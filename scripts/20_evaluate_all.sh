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

# The official task YAML asks for max_tokens=32768 -- exactly Qwen2.5-0.5B's
# context window -- so max_ctx_len = max_length - max_gen_toks is 0 and the task
# as published cannot run on this model. The cap has to come down; the question
# is only how far.
#
# The official runs (moleculariq-eval's with_config branch, all 34 model
# configs) pass no override at all, so on vLLM they ended up with an effective
# ceiling of ~32.3k -- i.e. "generate until EOS". 28672 sits within ~11% of that
# while reserving 4096 tokens for the prompt.
#
# 4096 of prompt budget is ~6x the longest prompt actually observed: measured
# over 22,800 rendered prompts the distribution is median 436 / p99 556 /
# max 702 tokens. That margin matters because an overlong prompt is
# *left-truncated*, which would silently eat the front of the system prompt --
# much worse than clipping a response.
#
# Neither limit binds in practice: the policy emits ~83 tokens on average.
# Recorded in every eval manifest under `generation_overrides` and applied
# identically to all four models.
MAX_GEN_TOKS="${MAX_GEN_TOKS:-28672}"

# `auto` probes batch sizes against the full context window and can thrash a
# 80 GB card before settling. Pin it if you see repeated OOM warnings.
BATCH_SIZE="${BATCH_SIZE:-auto}"

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
    --batch-size "$BATCH_SIZE"
    --max-gen-toks "$MAX_GEN_TOKS"
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
