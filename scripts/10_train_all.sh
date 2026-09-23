#!/usr/bin/env bash
# Train the three single-task models, one after another.
#
#   bash scripts/10_train_all.sh                      # all three
#   bash scripts/10_train_all.sh grpo-count-r001      # just one
#
# Each experiment preflights first: batch arithmetic, dataset hash, prompt
# rendering and one real rollout. A preflight failure stops the whole script
# rather than burning the rest of the GPU budget on a broken config.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

# The GPU phases run without network: everything was staged by 01_prefetch.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

EXPERIMENTS=("$@")
if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
  EXPERIMENTS=(grpo-count-r001 grpo-index-r001 grpo-constraint-r001)
fi

for experiment in "${EXPERIMENTS[@]}"; do
  config="configs/experiments/${experiment}.yaml"
  echo
  echo "=============================================================="
  echo " $experiment"
  echo "=============================================================="
  python -m miqgrpo.train_grpo preflight --config "$config"
  python -m miqgrpo.train_grpo train --config "$config"
done

echo
echo "checkpoints:"
for experiment in "${EXPERIMENTS[@]}"; do
  echo "  runs/${experiment}/final"
done
