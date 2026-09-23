#!/usr/bin/env bash
# Run the whole official MolecularIQ benchmark on the base model and the three
# trained checkpoints.
#
#   bash scripts/20_evaluate_all.sh
#   BACKEND=hf bash scripts/20_evaluate_all.sh      # if vLLM is unavailable
#
# Four separate runs, four immutable result directories. This is the only stage
# that touches the benchmark, and it never feeds anything back into training.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

BACKEND="${BACKEND:-vllm}"
SUFFIX="${SUFFIX:-r001}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

evaluate () {
  local label="$1" model_path="$2" experiment="$3"
  echo
  echo "=============================================================="
  echo " benchmark: $label"
  echo "=============================================================="
  local args=(
    --run-id "miq-eval-${label}-${SUFFIX}"
    --model-path "$model_path"
    --label "$label"
    --backend "$BACKEND"
  )
  if [ -n "$experiment" ]; then
    args+=(--experiment "$experiment")
  fi
  python -m miqgrpo.evaluate run "${args[@]}"
}

evaluate baseline              "$BASE_MODEL"                    ""
evaluate count                 "runs/grpo-count-r001/final"      grpo-count-r001
evaluate index                 "runs/grpo-index-r001/final"      grpo-index-r001
evaluate constraint_generation "runs/grpo-constraint-r001/final" grpo-constraint-r001

echo
echo "results in results/moleculariq/"
