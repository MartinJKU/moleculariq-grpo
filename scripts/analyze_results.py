#!/usr/bin/env python3
"""Per-task-type breakdown and paired significance tests against the baseline.

    python scripts/analyze_results.py

The headline metrics in each ``summary.json`` come from the official harness and
are reported as-is. This adds the two things a write-up needs on top of them:

1. **Where** a specialist improved -- overall accuracy hides whether the
   count-trained model got better at counting or just at formatting.
2. **Whether the difference is real.** With ~2% accuracy and 5,111 items, a
   +0.3 percentage point gap is inside the noise and a +0.9 one may not be.
   Every model answers the *same* items, so the comparison is paired: bootstrap
   over items rather than assuming independence, which is both correct and
   considerably more powerful than comparing two independent proportions.

It also reports response lengths correctly. ``resps`` is ``[[r1, r2, r3]]`` --
one element holding the three repeats -- so a naive ``len(str(resp))`` measures
the stringified triple and overstates the per-response length by ~3x.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

METRIC = "avg_accuracy"
TASK_LABELS = {
    "count": "counting",
    "index": "index attribution",
    "generation": "constrained generation",
}


def newest_samples(run_dir: Path) -> list[dict[str, Any]]:
    """Read only the newest sample file; a directory may hold several."""
    files = sorted(run_dir.rglob("samples_*.jsonl"))
    if not files:
        return []
    if len(files) > 1:
        print(
            f"  ! {run_dir.name}: {len(files)} sample files, using newest "
            f"({files[-1].name})"
        )
    records = []
    with open(files[-1]) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def flatten_responses(record: dict[str, Any]) -> list[str]:
    """Pull out the individual repeats, whatever nesting the harness used."""
    raw = record.get("filtered_resps") or record.get("resps") or []
    out: list[str] = []
    stack = list(raw)
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
        elif item is not None:
            out.append(item if isinstance(item, str) else str(item))
    return out


def key_of(record: dict[str, Any]) -> Any:
    doc = record.get("doc") or {}
    return record.get("doc_id", doc.get("uid"))


def task_of(record: dict[str, Any]) -> str:
    return str((record.get("doc") or {}).get("task_type", "unknown"))


def bootstrap_ci(
    diffs: list[float], iterations: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap over items. Returns (mean, lo, hi) at 95%."""
    if not diffs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(diffs)
    mean = sum(diffs) / n
    means = []
    for _ in range(iterations):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return mean, means[int(iterations * 0.025)], means[int(iterations * 0.975)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(os.environ.get("MIQ_RESULTS", "results")) / "moleculariq",
    )
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    runs: dict[str, list[dict[str, Any]]] = {}
    headline: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(args.results.glob("*/")):
        manifest = run_dir / "eval_manifest.json"
        summary = run_dir / "summary.json"
        if not summary.exists():
            continue
        meta = json.loads(manifest.read_text()) if manifest.exists() else {}
        if not meta.get("full_benchmark", True):
            print(f"  skipping {run_dir.name} (not a whole-benchmark run)")
            continue
        label = json.loads(summary.read_text()).get("label") or run_dir.name
        runs[label] = newest_samples(run_dir)
        headline[label] = json.loads(summary.read_text()).get("metrics", {})

    if args.baseline_label not in runs:
        raise SystemExit(f"no '{args.baseline_label}' run found in {args.results}")

    print("\n" + "=" * 78)
    print(" official headline metrics (from the harness, reported as-is)")
    print("=" * 78)
    for label, metrics in headline.items():
        bits = "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        print(f"  {label:24s} {bits}")

    print("\n" + "=" * 78)
    print(" response length per repeat (cap was max_gen_toks=28672)")
    print("=" * 78)
    for label, records in runs.items():
        lens = sorted(len(r) for rec in records for r in flatten_responses(rec))
        if not lens:
            continue
        n = len(lens)
        print(
            f"  {label:24s} n={n:6d}  median={lens[n // 2]:6d}  "
            f"p99={lens[int(n * 0.99)]:6d}  max={lens[-1]:7d} chars"
        )
    print("  (28672 tokens is roughly 90000-115000 chars; well under = cap never bound)")

    base_by_key = {key_of(r): r for r in runs[args.baseline_label]}

    print("\n" + "=" * 78)
    print(f" {METRIC} by task type")
    print("=" * 78)
    per_task: dict[str, dict[str, float]] = {}
    for label, records in runs.items():
        buckets: dict[str, list[float]] = defaultdict(list)
        for record in records:
            value = record.get(METRIC)
            if isinstance(value, (int, float)):
                buckets[task_of(record)].append(float(value))
        per_task[label] = {k: sum(v) / len(v) for k, v in buckets.items() if v}

    tasks = sorted({t for d in per_task.values() for t in d})
    header = "  " + "model".ljust(24) + "".join(
        TASK_LABELS.get(t, t)[:20].rjust(22) for t in tasks
    )
    print(header)
    for label in runs:
        row = "  " + label.ljust(24)
        for task in tasks:
            row += f"{per_task[label].get(task, float('nan')) * 100:21.2f}%"
        print(row)

    print("\n" + "=" * 78)
    print(f" paired change vs {args.baseline_label}, 95% bootstrap CI over items")
    print("=" * 78)
    print("  a CI that spans 0 means the difference is not distinguishable from noise\n")

    for label, records in runs.items():
        if label == args.baseline_label:
            continue
        print(f"  --- {label} ---")
        overall: list[float] = []
        by_task: dict[str, list[float]] = defaultdict(list)
        for record in records:
            key = key_of(record)
            base = base_by_key.get(key)
            if base is None:
                continue
            mine, theirs = record.get(METRIC), base.get(METRIC)
            if not isinstance(mine, (int, float)) or not isinstance(theirs, (int, float)):
                continue
            diff = float(mine) - float(theirs)
            overall.append(diff)
            by_task[task_of(record)].append(diff)

        for name, diffs in [("OVERALL", overall)] + [
            (TASK_LABELS.get(t, t), by_task[t]) for t in sorted(by_task)
        ]:
            mean, lo, hi = bootstrap_ci(diffs, args.bootstrap)
            verdict = "significant" if (lo > 0 or hi < 0) else "not significant"
            print(
                f"    {name:24s} {mean * 100:+6.2f} pp  "
                f"[{lo * 100:+6.2f}, {hi * 100:+6.2f}]  n={len(diffs):5d}  {verdict}"
            )
        print()

    print("=" * 78)
    print(" note: 'pp' is percentage points of avg_accuracy. Every model answered")
    print(" the same items, so differences are paired; the bootstrap resamples")
    print(" items, which is the unit of independent variation here.")
    print("=" * 78)


if __name__ == "__main__":
    main()
