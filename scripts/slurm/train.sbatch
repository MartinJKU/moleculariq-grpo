#!/bin/bash
# Phase 2: one single-task GRPO run. One GPU is the minimum that exercises the
# whole workflow; scale only after the batch arithmetic has been revalidated.
#
#   EXPERIMENT=grpo-count-r001 sbatch --account=<project> scripts/slurm/train.sbatch
#
#SBATCH --job-name=miq-grpo
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$WORK/moleculariq-grpo}"
VENV_PATH="${VENV_PATH:-$WORK/venvs/moleculariq-grpo}"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT=grpo-count-r001 | grpo-index-r001 | grpo-constraint-r001}"

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export MIQ_DATA="${MIQ_DATA:-$WORK/moleculariq-grpo-data}"
export MIQ_RUNS="${MIQ_RUNS:-$WORK/moleculariq-grpo-runs}"
export MIQ_RESULTS="${MIQ_RESULTS:-$WORK/moleculariq-grpo-results}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
source "$VENV_PATH/bin/activate"

echo "job=${SLURM_JOB_ID:-} host=$(hostname) experiment=$EXPERIMENT"
nvidia-smi

# Exactly one launcher. srun already starts the task; do not add torchrun or
# accelerate launch on top, or the world size doubles and the GRPO batch
# arithmetic silently stops meaning what the config says.
srun python -m miqgrpo.train_grpo preflight --config "configs/experiments/${EXPERIMENT}.yaml"
srun python -m miqgrpo.train_grpo train --config "configs/experiments/${EXPERIMENT}.yaml"
