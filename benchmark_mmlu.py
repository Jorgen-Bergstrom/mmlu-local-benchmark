#!/usr/bin/env python3
"""Benchmark local LLMs (served by llama.cpp) on the MMLU test set.

For each model this script:
  1. starts llama-server with the model's launch configuration
  2. answers every MMLU test question with a standard 5-shot prompt
     (examples from mmlu/data/dev/, questions from mmlu/data/test/)
  3. records per-question predictions plus per-subject and overall accuracy
  4. stops the server

Answers are generated greedily (temperature 0) and constrained by a GBNF
grammar to a single letter (A/B/C/D), so scoring is exact-match and
reproducible.

Examples:
  python3 benchmark_mmlu.py --list
  python3 benchmark_mmlu.py --model qwen3.5-9B --dry-run
  python3 benchmark_mmlu.py --model qwen3.5-9B --limit 10     # smoke test
  python3 benchmark_mmlu.py --model qwen3.5-9B                # full run
  ./run_all_models.sh                                          # all models
"""

import argparse
import csv
import json
import shlex
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("This script requires the 'requests' package (pip install requests)")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MMLU_DATA_DIR = PROJECT_DIR / "mmlu" / "data"
RESULTS_DIR = PROJECT_DIR / "results"

LLAMA_SERVER = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"
GGUF_CACHE = Path.home() / ".cache" / "llama.cpp"

# GBNF grammar: the response is exactly one letter, nothing else.
GRAMMAR = 'root ::= "A" | "B" | "C" | "D"'
LETTERS = ("A", "B", "C", "D")

REQUEST_TIMEOUT = (10, 180)  # (connect, read) seconds
REQUEST_RETRIES = 3

# --------------------------------------------------------------------------
# Model registry — launch configs inherited from the aliases in ~/.zshrc
# (model-loading args only; sampling defaults from the aliases are chat
# defaults and are overridden per-request by this benchmark)
# --------------------------------------------------------------------------

MODEL_REGISTRY = {
    "qwen3.6-35B-A3B": {
        "gguf": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "server_args": "-ngl 24 -ctk q4_0 -ctv q4_0 --n-cpu-moe 18 -fa on --jinja -c 131072",
        "notes": "alias qwen3.6_35B_A3B_128k",
    },
    "qwen3.8-27B-IQ3_S": {
        "gguf": "Qwen3.8-27B-UD-IQ3_S.gguf",
        "server_args": "-ngl 99 -ctk q4_0 -ctv q4_0 -fa on --jinja -c 32768",
        "notes": "alias qwen3.8_27B | BENCH OVERRIDE: ctx 32768 (8192/slot) + full GPU offload (alias used -c 131072 / -ngl 24)",
    },
    "qwen3.8-27B-IQ3_XXS": {
        "gguf": "Qwen3.8-27B-UD-IQ3_XXS.gguf",
        "server_args": "-ngl 99 -ctk q4_0 -ctv q4_0 -fa on --jinja -c 32768",
        "notes": "INFERRED from qwen3.8-27B-IQ3_S | BENCH OVERRIDE: ctx 32768 (8192/slot) + full GPU offload (was -c 131072 / -ngl 24)",
    },
    "qwen3-coder-30B-A3B": {
        "gguf": "Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf",
        "server_args": "-ngl 99 -ctk q8_0 -ctv q8_0 --n-cpu-moe 16 -fa on --jinja -c 32768",
        "notes": "alias qwen3_Coder_30B_A3B | BENCH OVERRIDE: ctx 32768 (8192/slot) + full GPU offload (was -c 16384 / -ngl 36)",
    },
    "gemma4-12B": {
        "gguf": "gemma-4-12b-it-Q4_K_M.gguf",
        "server_args": "-ngl 99 -ctk q4_0 -ctv q4_0 -c 32768 -fa on --jinja",
        "notes": "alias gemma4_12B | BENCH OVERRIDE: ctx 32768 (8192/slot) + full GPU offload (was -c 262144 / -ngl 49)",
    },
    "gemma4-26B-A4B": {
        "gguf": "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
        "server_args": "-ngl 99 -ctk q8_0 -ctv q8_0 --n-cpu-moe 16 -fa on --jinja -c 32768",
        "notes": "alias gemma4_26B_A3B | BENCH OVERRIDE: ctx 32768 (8192/slot) + full GPU offload (was -c 16384 / -ngl 36)",
    },
    "qwen3.5-9B": {
        "gguf": "unsloth_Qwen3.5-9B-GGUF_Qwen3.5-9B-Q4_K_M.gguf",
        "server_args": "-ngl 99 -ctk q8_0 -ctv q8_0 -fa on --jinja -c 32768",
        "notes": "alias qwen3.5_9B | BENCH OVERRIDE: ctx 32768 (8192/slot) + full GPU offload (was -c 16384 / -ngl 24). "
                 "The mmproj file for this model is not needed for this text-only benchmark",
    },
    "qwen3.5-35B-A3B": {
        "gguf": "unsloth_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf",
        "server_args": "-ngl 99 -ctk q8_0 -ctv q8_0 --n-cpu-moe 16 -fa on --jinja -c 32768",
        "notes": "from the commented-out alias qwen3.5_35B_A3B | BENCH OVERRIDE: ctx 32768 (8192/slot) + full GPU offload (was -c 32768 / -ngl 24). "
                 "The mmproj file for this model is not needed for this text-only benchmark",
    },
}

# --------------------------------------------------------------------------
# MMLU data
# --------------------------------------------------------------------------


def load_split(split):
    """Return {subject: [[question, A, B, C, D, answer], ...]} for a split.

    Uses the csv module because fields may contain commas, quotes and
    newlines.
    """
    split_dir = MMLU_DATA_DIR / split
    if not split_dir.is_dir():
        sys.exit(f"MMLU {split} split not found: {split_dir}")
    suffix = f"_{split}.csv"
    data = {}
    for path in sorted(split_dir.glob(f"*{suffix}")):
        subject = path.name[: -len(suffix)]
        with path.open(newline="", encoding="utf-8") as f:
            data[subject] = [row for row in csv.reader(f) if row]
    return data


def build_prompt(subject, dev_rows, test_row, shots):
    """Standard MMLU few-shot prompt for one test question."""
    subj = subject.replace("_", " ")
    lines = [f"The following are multiple choice questions (with answers) about {subj}.", ""]
    for row in dev_rows[:shots]:
        q, a, b, c, d, ans = row
        lines += [f"Q: {q}", f"A. {a}", f"B. {b}", f"C. {c}", f"D. {d}", f"Answer: {ans}", ""]
    q, a, b, c, d, _gold = test_row
    lines += [f"Q: {q}", f"A. {a}", f"B. {b}", f"C. {c}", f"D. {d}", "Answer:"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# llama-server lifecycle
# --------------------------------------------------------------------------


class LlamaServer:
    def __init__(self, cmd, host, port, log_path, expected_model=None):
        self.cmd = cmd
        self.host = host
        self.port = port
        self.log_path = Path(log_path)
        self.expected_model = Path(expected_model) if expected_model else None
        self.proc = None
        self._log_file = None

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def _port_in_use(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((self.host, self.port)) == 0

    def start(self):
        # If something already listens on the port, our llama-server would fail
        # to bind, exit, and the benchmark would silently send every request to
        # the pre-existing server (i.e. benchmark the wrong model). Fail loudly
        # instead.
        if self._port_in_use():
            raise RuntimeError(
                f"Something is already listening on {self.host}:{self.port}; "
                f"refusing to start. Stop that process first (or pass --port to "
                f"use a free port). Otherwise llama-server cannot bind and the "
                f"benchmark would silently query the existing server."
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w", buffering=1)
        # start_new_session: the server survives our Ctrl+C and is stopped
        # explicitly via stop()
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _verify_serving_expected_model(self):
        """Ensure the server answering on our port is the one we started.

        Extra guard for the race where our llama-server fails to bind and an
        already-running server (a different model) answers /health first.
        """
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited (code {self.proc.returncode}) while "
                f"starting; see {self.log_path}"
            )
        if self.expected_model is None:
            return
        try:
            r = requests.get(f"{self.base_url}/props", timeout=5)
            r.raise_for_status()
            served = r.json().get("model_path")
        except requests.RequestException as e:
            raise RuntimeError(
                f"Could not verify the model served at {self.base_url}: {e}"
            )
        if served and Path(served).name != self.expected_model.name:
            raise RuntimeError(
                f"Server at {self.base_url} is serving {Path(served).name!r}, "
                f"but expected {self.expected_model.name!r}; refusing to run "
                f"against the wrong model."
            )

    def wait_ready(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (code {self.proc.returncode}); "
                    f"see {self.log_path}"
                )
            try:
                r = requests.get(f"{self.base_url}/health", timeout=5)
                if r.ok:
                    self._verify_serving_expected_model()
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(
            f"llama-server not ready after {timeout_s}s; see {self.log_path}"
        )

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self._log_file is not None:
            self._log_file.close()


# --------------------------------------------------------------------------
# Benchmark client
# --------------------------------------------------------------------------


def ask_once(base_url, prompt):
    """POST one completion request; retries transient errors."""
    payload = {
        "prompt": prompt,
        "grammar": GRAMMAR,
        "temperature": 0,        # greedy
        "max_tokens": 1,         # current llama-server name
        "n_predict": 1,          # older llama-server name (harmless if ignored)
        "n": 1,
    }
    last_exc = None
    for attempt in range(REQUEST_RETRIES):
        try:
            r = requests.post(f"{base_url}/completion", json=payload,
                              timeout=REQUEST_TIMEOUT)
            if r.status_code == 400:
                # bad request (e.g. unsupported option) - retrying won't help
                raise RuntimeError(f"HTTP 400 from llama-server: {r.text[:300]}")
            r.raise_for_status()
            return r.json().get("content", "")
        except requests.RequestException as e:
            last_exc = e
            if attempt < REQUEST_RETRIES - 1:
                time.sleep(min(2 ** attempt, 30))
    raise last_exc


def parse_letter(content):
    text = content.strip()
    if text in LETTERS:
        return text
    for ch in text:  # fallback: first letter anywhere in the response
        if ch in LETTERS:
            return ch
    return None


def process_one(base_url, subject, idx, test_row, dev_rows, shots):
    qid = f"{subject}:{idx}"
    question, a, b, c, d, gold = test_row
    record = {
        "id": qid,
        "subject": subject,
        "question": question,
        "choices": [a, b, c, d],
        "gold": gold,
    }
    prompt = build_prompt(subject, dev_rows, test_row, shots)
    t0 = time.monotonic()
    try:
        content = ask_once(base_url, prompt)
        pred = parse_letter(content)
        record["latency_s"] = round(time.monotonic() - t0, 3)
        if pred is None:
            record["pred"] = None
            record["correct"] = False
            record["error"] = f"unparseable response: {content!r}"
        else:
            record["pred"] = pred
            record["correct"] = (pred == gold)
    except Exception as e:
        record["pred"] = None
        record["correct"] = False
        record["error"] = f"{type(e).__name__}: {e}"
        record["latency_s"] = round(time.monotonic() - t0, 3)
    return record


# --------------------------------------------------------------------------
# Aggregation and results
# --------------------------------------------------------------------------


def aggregate(results, scope):
    per_subject = {}
    for subject in sorted({s for s, _ in scope}):
        recs = [results[f"{subject}:{i}"] for s, i in scope
                if s == subject and f"{subject}:{i}" in results]
        answered = [r for r in recs if "error" not in r]
        correct = sum(1 for r in answered if r["correct"])
        per_subject[subject] = {
            "total": len(recs),
            "answered": len(answered),
            "failed": len(recs) - len(answered),
            "correct": correct,
            "accuracy": (correct / len(answered)) if answered else None,
        }
    total_answered = sum(v["answered"] for v in per_subject.values())
    total_correct = sum(v["correct"] for v in per_subject.values())
    total_failed = sum(v["failed"] for v in per_subject.values())
    accs = [v["accuracy"] for v in per_subject.values() if v["accuracy"] is not None]
    return {
        "total_questions": len(scope),
        "total_answered": total_answered,
        "total_failed": total_failed,
        "total_correct": total_correct,
        "overall_accuracy": (total_correct / total_answered) if total_answered else None,
        "macro_accuracy": (sum(accs) / len(accs)) if accs else None,
        "per_subject": per_subject,
    }


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _pct(x):
    return f"{x * 100:.1f}%" if isinstance(x, float) else "n/a"


def update_summaries(run_info):
    summary_path = RESULTS_DIR / "summary.json"
    entries = []
    if summary_path.exists():
        entries = json.loads(summary_path.read_text(encoding="utf-8"))
    entries = [e for e in entries if e.get("model") != run_info["model"]]
    entries.append(run_info)
    summary_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    lines = [
        "# MMLU benchmark results - local LLMs (llama.cpp)",
        "",
        "Greedy decoding, 5-shot, grammar-constrained single-letter answers.",
        "",
        "| Model | Overall | Macro | Correct/Answered | Failed | Q/s | Date |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        lines.append(
            f"| {e.get('model')} | {_pct(e.get('overall_accuracy'))} | "
            f"{_pct(e.get('macro_accuracy'))} | "
            f"{e.get('total_correct', 0)}/{e.get('total_answered', 0)} | "
            f"{e.get('total_failed', 0)} | {e.get('questions_per_second')} | "
            f"{str(e.get('finished_at', ''))[:10]} |"
        )
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Main flow
# --------------------------------------------------------------------------


def run_benchmark(args, model_key, model_path, server_args, notes):
    if not model_path.exists():
        sys.exit(f"Model file not found: {model_path}")

    run_dir = RESULTS_DIR / model_key
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"
    log_path = run_dir / "server.log"

    server_cmd = [
        str(LLAMA_SERVER),
        "-m", str(model_path),
        "--host", args.host,
        "--port", str(args.port),
        "-np", str(args.concurrency),
    ] + server_args

    if args.dry_run:
        print("Server command (not started):")
        print("  " + " ".join(server_cmd))
        print(f"Benchmark: MMLU test split, {args.shots}-shot, "
              f"concurrency {args.concurrency}"
              + (f", limit {args.limit}/subject" if args.limit else ""))
        print(f"Results would be written to: {run_dir}")
        return

    test_data = load_split("test")
    dev_data = load_split("dev")

    # Scope: (subject, question index) pairs for this run
    scope = []
    for subject, rows in test_data.items():
        n = args.limit if args.limit else len(rows)
        for idx in range(min(n, len(rows))):
            scope.append((subject, idx))

    # Resume support: preload previously answered questions
    results = {}
    resume = pred_path.exists() and not args.fresh
    if resume:
        with pred_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    results[rec["id"]] = rec

    todo = [(s, i) for s, i in scope if f"{s}:{i}" not in results]
    print(f"Model: {model_key} ({model_path.name})")
    print(f"MMLU questions in scope: {len(scope)} "
          f"(already answered: {len(scope) - len(todo)}, to run: {len(todo)})")
    if not todo:
        print("Nothing to run - all questions in scope already answered "
              "(use --fresh to ignore previous results)")

    started_at = datetime.now(timezone.utc).isoformat()
    server = LlamaServer(server_cmd, args.host, args.port, log_path,
                         expected_model=model_path)
    server.start()
    print(f"Started llama-server (pid {server.proc.pid}), waiting for model to load...")
    try:
        server.wait_ready(args.startup_timeout)
        print(f"Server ready at {server.base_url}")

        elapsed = 0.0
        if todo:
            t0 = time.monotonic()
            done = 0
            lock = threading.Lock()
            pred_file = open(pred_path, "a" if resume else "w", encoding="utf-8")
            try:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures = {
                        pool.submit(process_one, server.base_url, s, i,
                                    test_data[s][i], dev_data.get(s, []), args.shots): (s, i)
                        for s, i in todo
                    }
                    for fut in as_completed(futures):
                        rec = fut.result()
                        with lock:
                            results[rec["id"]] = rec
                            pred_file.write(json.dumps(rec) + "\n")
                            pred_file.flush()
                            done += 1
                            if done % 200 == 0 or done == len(todo):
                                print(f"\r  progress: {done}/{len(todo)}",
                                      end="", flush=True)
                print()
            finally:
                pred_file.close()
            elapsed = time.monotonic() - t0
    finally:
        server.stop()
        print("llama-server stopped")

    agg = aggregate(results, scope)
    print("\nPer-subject accuracy:")
    for subject, st in agg["per_subject"].items():
        print(f"  {subject:32s} {_pct(st['accuracy']):>6s}  "
              f"({st['correct']}/{st['answered']})")
    print(f"Overall accuracy: {_pct(agg['overall_accuracy'])}  "
          f"Macro: {_pct(agg['macro_accuracy'])}  "
          f"Failed: {agg['total_failed']}")

    run_info = {
        "model": model_key,
        "gguf": str(model_path),
        "notes": notes,
        "server_command": server_cmd,
        "config": {
            "shots": args.shots,
            "concurrency": args.concurrency,
            "limit_per_subject": args.limit,
            "temperature": 0,
            "grammar": GRAMMAR,
            "endpoint": "/completion",
            "host": args.host,
            "port": args.port,
        },
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "questions_per_second": round(len(todo) / elapsed, 3) if elapsed > 0 else None,
        "gpu": gpu_info(),
        **agg,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8")
    update_summaries(run_info)
    print(f"Results written to {run_dir / 'run.json'}")
    print(f"Cross-model summary updated: {RESULTS_DIR / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark local LLMs (llama.cpp llama-server) on the MMLU test set.")
    ap.add_argument("--model", help="model key from the registry (see --list)")
    ap.add_argument("--model-file",
                    help="path to a GGUF file not in the registry (use with --server-args)")
    ap.add_argument("--server-args", default="",
                    help="llama-server args for --model-file, e.g. \"-ngl 24 -c 16384\"")
    ap.add_argument("--list", action="store_true", help="list registered models")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact server command without starting anything")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="parallel requests / server slots (default: 4)")
    ap.add_argument("--limit", type=int, default=None,
                    help="questions per subject (for smoke tests; default: all)")
    ap.add_argument("--shots", type=int, default=5, choices=range(1, 6),
                    help="few-shot examples from the dev split (default: 5; dev has exactly 5)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface for llama-server (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--startup-timeout", type=int, default=1200,
                    help="seconds to wait for the model to load (default: 1200)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore previously answered questions and overwrite predictions")
    args = ap.parse_args()

    if not LLAMA_SERVER.exists():
        sys.exit(f"llama-server not found at {LLAMA_SERVER}")
    if args.concurrency < 1:
        ap.error("--concurrency must be >= 1")

    if args.list:
        for key, cfg in MODEL_REGISTRY.items():
            print(f"{key:22s} {GGUF_CACHE / cfg['gguf']}")
            print(f"{'':22s} server args: {cfg['server_args']}")
            print(f"{'':22s} note: {cfg['notes']}")
        return

    if args.model:
        if args.model not in MODEL_REGISTRY:
            sys.exit(f"Unknown model {args.model!r} (see --list)")
        cfg = MODEL_REGISTRY[args.model]
        model_key, model_path = args.model, GGUF_CACHE / cfg["gguf"]
        server_args, notes = cfg["server_args"].split(), cfg["notes"]
    elif args.model_file:
        model_path = Path(args.model_file).resolve()
        model_key = model_path.stem
        server_args = shlex.split(args.server_args)
        notes = "custom --model-file run"
    else:
        ap.error("one of --model or --model-file is required")

    try:
        run_benchmark(args, model_key, model_path, server_args, notes)
    except KeyboardInterrupt:
        print("\nInterrupted; server stopped. Re-run to resume where it left off.")
        sys.exit(130)


if __name__ == "__main__":
    main()
