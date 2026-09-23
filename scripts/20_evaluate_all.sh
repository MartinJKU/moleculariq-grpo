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

# Generation kwargs, passed through to lm_eval. Every value here exists to make
# the HF backend behave like the official vLLM runs -- none of it is a choice
# about how the model should answer. See docs/official-semantics.md.
#
# max_gen_toks=28672
#   The task YAML asks for max_tokens=32768, exactly Qwen2.5-0.5B's context
#   window, so max_ctx_len = max_length - max_gen_toks = 0 and the run asserts.
#   The official vLLM runs survived the identical arithmetic only because
#   truncate_tokens does tokens[-0:], which in Python returns the whole list --
#   leaving them an effective ceiling of ~32.3k. 28672 is within ~11% of that
#   and reserves 4096 tokens for the prompt, ~6x the longest prompt measured
#   (median 436 / p99 556 / max 702 over 22,800 rendered prompts).
#
# temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0
#   The task sets do_sample=true and no temperature. The vLLM backend drops
#   do_sample and never sets temperature, so vLLM's SamplingParams defaults
#   applied: temperature 1.0, top_p 1.0, top_k off, no repetition penalty.
#   The HF backend instead injects temperature=0.0 (huggingface.py:996) and
#   then raises on do_sample=true. Even once that is fixed, leaving these unset
#   makes HF fall back to Qwen2.5-0.5B-Instruct's own generation_config
#   (0.7 / 0.8 / 20 / 1.1) -- different sampling from the official runs, and a
#   silent difference rather than a loud one. In HF, top_k=0 and top_p=1.0
#   disable those warpers, and repetition_penalty=1.0 is a no-op, so this
#   reproduces plain temperature-1.0 sampling.
#
# All of it is recorded in each eval manifest under `generation_overrides` and
# applied identically to all four models.
GEN_KWARGS="${GEN_KWARGS:-max_gen_toks=28672,temperature=1.0,top_p=1.0,top_k=0,repetition_penalty=1.0}"

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
    --gen-kwargs "$GEN_KWARGS"
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
