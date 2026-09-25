#!/usr/bin/env bash
# Collect everything a write-up needs into one small archive.
#
#   bash scripts/40_bundle_results.sh              # figures, metrics, manifests
#   bash scripts/40_bundle_results.sh --with-raw   # + per-sample logs (large)
#
# Prints the archive's size and checksum, and the exact command to pull it to a
# local machine -- including the ssh-pipe form, which works on hosts whose SSH
# endpoint has no scp support.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${MIQ_DATA:-$PROJECT_DIR/data}"
RUNS="${MIQ_RUNS:-$PROJECT_DIR/runs}"
RESULTS="${MIQ_RESULTS:-$PROJECT_DIR/results}"
OUT_DIR="${OUT_DIR:-$(dirname "$RESULTS")}"

WITH_RAW=0
[ "${1:-}" = "--with-raw" ] && WITH_RAW=1

echo "reading from:"
echo "  data     $DATA"
echo "  runs     $RUNS"
echo "  results  $RESULTS"
if [ -z "${MIQ_RESULTS:-}" ]; then
  echo
  echo "  ! MIQ_RESULTS is not set in this environment; the paths above are"
  echo "    repo-relative defaults and probably not where your results are."
  echo "    Run 'source ~/.bashrc' or pass them inline."
fi
echo

STAGE="$OUT_DIR/miq-bundle"
rm -rf "$STAGE"
mkdir -p "$STAGE/evaluations" "$STAGE/training" "$STAGE/dataset"

copied=0
missing=0
take () {  # take <src> <dest-dir>
  if [ -e "$1" ]; then
    cp -r "$1" "$2" && copied=$((copied + 1))
  else
    echo "  missing: $1"
    missing=$((missing + 1))
  fi
}

echo "collecting:"
take "$RESULTS/figures"      "$STAGE/"
take "$RESULTS/analysis.txt" "$STAGE/"

for d in "$RESULTS"/moleculariq/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  mkdir -p "$STAGE/evaluations/$name"
  for f in summary.json eval_manifest.json eval_manifest.yaml provenance.json environment.txt; do
    [ -e "$d/$f" ] && cp "$d/$f" "$STAGE/evaluations/$name/" && copied=$((copied + 1))
  done
  if [ "$WITH_RAW" = "1" ]; then
    # Only the newest sample file: a directory can hold several from reruns,
    # and mixing them corrupts any per-item analysis.
    newest=$(ls -1 "$d"raw/**/samples_*.jsonl "$d"raw/samples_*.jsonl 2>/dev/null | sort | tail -1 || true)
    if [ -n "$newest" ]; then
      cp "$newest" "$STAGE/evaluations/$name/" && copied=$((copied + 1))
    fi
    newest_results=$(ls -1 "$d"raw/**/results_*.json "$d"raw/results_*.json 2>/dev/null | sort | tail -1 || true)
    [ -n "$newest_results" ] && cp "$newest_results" "$STAGE/evaluations/$name/" && copied=$((copied + 1))
  fi
done

for d in "$RUNS"/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  mkdir -p "$STAGE/training/$name"
  for f in frozen_config.yaml provenance.json training_summary.json; do
    [ -e "$d/$f" ] && cp "$d/$f" "$STAGE/training/$name/" && copied=$((copied + 1))
  done
  [ -e "$d/logs/metrics.jsonl" ] && cp "$d/logs/metrics.jsonl" "$STAGE/training/$name/" && copied=$((copied + 1))
done

for artifact in "$DATA"/processed/*/; do
  [ -d "$artifact" ] || continue
  name=$(basename "$artifact")
  mkdir -p "$STAGE/dataset/$name"
  for f in manifest.json preprocessing_config.yaml provenance.json; do
    [ -e "$artifact/$f" ] && cp "$artifact/$f" "$STAGE/dataset/$name/" && copied=$((copied + 1))
  done
done

[ -e "$PROJECT_DIR/assets_manifest.json" ] && cp "$PROJECT_DIR/assets_manifest.json" "$STAGE/" && copied=$((copied + 1))

# The exact code revision these results came from.
{
  echo "commit: $(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch: $(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "dirty:  $(git -C "$PROJECT_DIR" status --porcelain 2>/dev/null | wc -l) modified files"
  echo
  git -C "$PROJECT_DIR" status --porcelain 2>/dev/null || true
  echo
  git -C "$PROJECT_DIR" diff HEAD 2>/dev/null || true
} > "$STAGE/code_revision.txt"

ARCHIVE="$OUT_DIR/miq-report-bundle.tgz"
rm -f "$ARCHIVE"
tar czf "$ARCHIVE" -C "$OUT_DIR" miq-bundle

echo
echo "=============================================================="
echo "  files collected : $copied"
[ "$missing" -gt 0 ] && echo "  missing         : $missing (listed above)"
echo "  archive         : $ARCHIVE"
echo "  size            : $(du -h "$ARCHIVE" | cut -f1)"
# sha256sum on Linux, shasum on macOS -- the archive gets checked on both ends.
if command -v sha256sum >/dev/null 2>&1; then
  echo "  sha256          : $(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
else
  echo "  sha256          : $(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)"
fi
echo "=============================================================="
echo
echo "Pull it to your local machine. If your SSH endpoint has no scp support,"
echo "pipe it instead (-T matters: a pty would corrupt the binary):"
echo
echo "  ssh -T <your-ssh-target> \"cat $ARCHIVE\" > ~/Downloads/$(basename "$ARCHIVE")"
echo
echo "then verify the checksum matches on both ends."
