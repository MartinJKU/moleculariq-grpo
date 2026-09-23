#!/bin/bash
# Phase 3: the whole official benchmark for one model. Separate job, separate
# result directory, never chained onto training automatically.
#
#   LABEL=count MODEL_PATH=$MIQ_RUNS/grpo-count-r001/final EXPERIMENT=grpo-count-r001 \
#     sbatch --account=<project> scripts/slurm/evaluate.sbatch
#
#SBATCH --job-name=miq-eval
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$WORK/moleculariq-grpo}"
VENV_PATH="${VENV_PATH:-$WORK/venvs/moleculariq-grpo}"
LABEL="${LABEL:?set LABEL=baseline|count|index|constraint_generation}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
EXPERIMENT="${EXPERIMENT:-}"
RUN_ID="${RUN_ID:-miq-eval-${LABEL}-r001}"
BACKEND="${BACKEND:-vllm}"

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export MIQ_DATA="${MIQ_DATA:-$WORK/moleculariq-grpo-data}"
export MIQ_RUNS="${MIQ_RUNS:-$WORK/moleculariq-grpo-runs}"
export MIQ_RESULTS="${MIQ_RESULTS:-$WORK/moleculariq-grpo-results}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
source "$VENV_PATH/bin/activate"

echo "job=${SLURM_JOB_ID:-} host=$(hostname) label=$LABEL"
nvidia-smi

ARGS=(run --run-id "$RUN_ID" --model-path "$MODEL_PATH" --label "$LABEL" --backend "$BACKEND")
if [ -n "$EXPERIMENT" ]; then
  ARGS+=(--experiment "$EXPERIMENT")
fi

srun python -m miqgrpo.evaluate "${ARGS[@]}"
