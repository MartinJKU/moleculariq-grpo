#!/bin/bash
# Phase 1: build the frozen dataset. CPU-bound; no GPU requested.
#
#   sbatch --account=<project> scripts/slurm/preprocess.sbatch
#
#SBATCH --job-name=miq-preprocess
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

# Partition and QoS policy on Leonardo changes; confirm with `sinfo` and
# `scontrol show partition <name>` before trusting the defaults above.

PROJECT_DIR="${PROJECT_DIR:-$WORK/moleculariq-grpo}"
VENV_PATH="${VENV_PATH:-$WORK/venvs/moleculariq-grpo}"
CONFIG="${CONFIG:-configs/preprocessing/miq-train-v001.yaml}"

# Everything persistent lives on $WORK: Booster /tmp is small and not kept.
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export MIQ_DATA="${MIQ_DATA:-$WORK/moleculariq-grpo-data}"
export MIQ_RUNS="${MIQ_RUNS:-$WORK/moleculariq-grpo-runs}"
export MIQ_RESULTS="${MIQ_RESULTS:-$WORK/moleculariq-grpo-results}"

cd "$PROJECT_DIR"
source "$VENV_PATH/bin/activate"

echo "job=${SLURM_JOB_ID:-} host=$(hostname)"
python -V

srun bash scripts/02_build_dataset.sh "$CONFIG"
