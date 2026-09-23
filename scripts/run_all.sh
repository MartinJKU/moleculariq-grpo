#!/usr/bin/env bash
# The whole pipeline, in order. Expects scripts/00_setup_env.sh to have run.
#
#   bash scripts/run_all.sh
#
#   dataset (CPU, ~10 min)
#     -> 3 x GRPO training (GPU, ~4-6 h each at the shipped settings)
#     -> 4 x official benchmark (GPU, ~1-2 h each with vLLM)
#     -> figures
#
# Stages are separate commands on purpose: each writes an immutable artifact,
# and a failure in one does not force redoing the others.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

bash scripts/02_build_dataset.sh
bash scripts/10_train_all.sh
bash scripts/20_evaluate_all.sh
bash scripts/30_make_plots.sh
