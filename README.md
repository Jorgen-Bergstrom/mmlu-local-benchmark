# mmlu-local-benchmark

Benchmark local GGUF LLMs on the official **MMLU** test set using
[`llama.cpp`](https://github.com/ggml-org/llama.cpp)'s `llama-server`.

For each model, in sequence, the harness starts `llama-server`, answers every
MMLU test question with a standard 5-shot prompt, records the predictions and
per-subject accuracy, stops the server, and moves on to the next model.

> The companion blog write-up lives on
> [bergstrom.org](https://bergstrom.org/posts/mmlu_local_llms/).

## Results

Greedy decoding, 5-shot, grammar-constrained single-letter answers, full
14,042-question test set per model, zero failed requests. Single
NVIDIA RTX 5060 Ti (16 GB).

| Model | Overall | Macro | Correct/Answered |
|---|---:|---:|---:|
| qwen3.6-35B-A3B | **84.1%** | 84.9% | 11813 / 14042 |
| qwen3.5-35B-A3B | **83.8%** | 84.5% | 11772 / 14042 |
| qwen3.8-27B-IQ3_S | 80.8% | 81.8% | 11342 / 14042 |
| qwen3.8-27B-IQ3_XXS | 79.3% | 81.1% | 11130 / 14042 |
| qwen3.5-9B | 79.0% | 80.1% | 11096 / 14042 |
| qwen3-coder-30B-A3B | 76.7% | 78.7% | 10771 / 14042 |
| gemma4-26B-A4B | 60.9% | 62.0% | 8550 / 14042 |
| gemma4-12B | 60.5% | 60.6% | 8493 / 14042 |

*Overall = micro accuracy. Macro = unweighted mean of the 57 per-subject
accuracies. Full per-subject numbers are in `results/<model>/run.json`; the
generated cross-model table is `results/summary.{md,json}`.*

These are **non-reasoning** scores: the harness constrains the output to a
single letter with a GBNF grammar and a one-token budget, so no model gets to
deliberate. Model names come from a local registry — your quantizations and
launch flags will differ.

## Benchmark configuration

- Decoding: greedy (`temperature = 0`)
- Answer format: GBNF grammar forces a single letter `A|B|C|D`; `max_tokens = 1`
- Prompts: standard 5-shot, examples from `mmlu/data/dev/<subject>_dev.csv`
- Scoring: exact letter match vs. the gold answer
- Concurrency: 4 parallel requests (server started with `-np 4`)
- Server bound to `127.0.0.1:8080` (configurable via `--host` / `--port`)

## Getting the MMLU data

The dataset is **not** committed (it is large and publicly available). Download
the canonical Berkeley tarball and extract it into `mmlu/`:

```bash
cd mmlu
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar          # creates mmlu/data/{test,dev,val,auxiliary_train}/
```

The `dev` split supplies the 5-shot examples; the `test` split is the scoring
set. See `mmlu/README.md` for the layout and CSV format. (Equivalently:
`load_dataset("cais/mmlu", "all")` on Hugging Face.)

## Running it

```bash
python3 benchmark_mmlu.py --list                          # list registered models
python3 benchmark_mmlu.py --model qwen3.5-9B --limit 10   # smoke test
python3 benchmark_mmlu.py --model qwen3.5-9B              # one model, full run
./run_all_models.sh                                       # every model, sequential
```

Useful flags and environment overrides:

- `--limit <n>` — questions per subject (smoke tests)
- `--concurrency <n>` — parallel requests / server slots (default 4)
- `--shots <n>` — few-shot examples from `dev/` (default 5)
- `--dry-run` — print the exact `llama-server` command, start nothing
- `--fresh` — ignore previous results and start over
- `run_all_models.sh`: `CONCURRENCY=4`, `LIMIT=<n>`, `MODELS="a b c"`

Runs are **resumable**: predictions stream to
`results/<model>/predictions.jsonl`, so a crash does not cost the whole run.

## Outputs

Per model, in `results/<model>/`:

- `run.json` — config, server command, overall + per-subject accuracy, timings, GPU
- `predictions.jsonl` — one line per question (id, subject, gold, pred, correct, latency)
- `server.log` — raw `llama-server` output

Cross-model:

- `results/summary.json` — array of per-model summaries
- `results/summary.md` — human-readable comparison table

The per-question `predictions.jsonl` and the raw `server.log` files are
gitignored (they are large). The full prediction sets for every model in the
table are attached, gzipped, to this repository's **Releases**.

## Requirements

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) built with `llama-server`
  (default path `~/llama.cpp/build/bin/llama-server`)
- Python 3 with the `requests` package
- The MMLU CSVs under `mmlu/data/` (see above)

Model launch configurations live in `benchmark_mmlu.py` under `MODEL_REGISTRY`
(GGUF file name + `llama-server` args per model); edit them there for your own
models or use `--model-file` with `--server-args`.

## License

MIT — see [`LICENSE`](LICENSE).

