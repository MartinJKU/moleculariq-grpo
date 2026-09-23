#!/usr/bin/env bash
# Build and verify the frozen training dataset. CPU only, ~10 minutes.
#
#   bash scripts/02_build_dataset.sh [config]
#
# Verification runs as a *separate* process on purpose: a dataset that only
# works inside the builder's memory is not a frozen artifact.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${1:-$PROJECT_DIR/configs/preprocessing/miq-train-v001.yaml}"
ARTIFACT_ID="$(python3 -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['dataset_artifact_id'])" "$CONFIG")"

cd "$PROJECT_DIR"
python -m miqgrpo.build_dataset build --config "$CONFIG"
python -m miqgrpo.build_dataset verify --artifact "$ARTIFACT_ID"
python scripts/03_fill_dataset_hash.py --artifact "$ARTIFACT_ID"

echo
echo "dataset '$ARTIFACT_ID' is frozen and pinned into the experiment configs."
