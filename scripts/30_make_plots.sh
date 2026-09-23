#!/usr/bin/env bash
# Build every report figure from whatever runs and results exist.
#
#   bash scripts/30_make_plots.sh
#
# Safe to run mid-project: missing pieces are skipped with a note rather than
# faked.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

python -m miqgrpo.plots all

echo
echo "figures in results/figures/"
