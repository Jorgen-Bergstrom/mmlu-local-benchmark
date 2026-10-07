#!/usr/bin/env zsh
# Run the MMLU benchmark for every local model, one at a time.
#
# For each model: start llama-server -> run the benchmark -> record results
# -> stop the server -> next model.
#
# Runs are resumable: re-running this script skips questions that are already
# recorded in results/<model>/predictions.jsonl.
#
# Environment overrides:
#   CONCURRENCY=4   parallel requests per model (default 4)
#   LIMIT=          questions per subject, e.g. LIMIT=10 for a smoke test
#   MODELS="a b c"  space-separated subset of model keys (default: all)

set -euo pipefail
cd "$(dirname "$0")"

ALL_MODELS=(
  qwen3.6-35B-A3B
  qwen3.8-27B-IQ3_S
  qwen3.8-27B-IQ3_XXS
  qwen3-coder-30B-A3B
  gemma4-12B
  gemma4-26B-A4B
  qwen3.5-9B
  qwen3.5-35B-A3B
)

CONCURRENCY="${CONCURRENCY:-4}"

if [[ -n "${MODELS:-}" ]]; then
  models=(${(s: :)MODELS})
else
  models=($ALL_MODELS)
fi

for m in $models; do
  echo "======================================================================"
  echo "[$(date)] === MODEL: $m ==="
  cmd=(python3 benchmark_mmlu.py --model "$m" --concurrency "$CONCURRENCY")
  if [[ -n "${LIMIT:-}" ]]; then
    cmd+=(--limit "$LIMIT")
  fi
  $cmd
  echo "[$(date)] === DONE: $m ==="
done

echo "All models complete. Cross-model summary: results/summary.md"
