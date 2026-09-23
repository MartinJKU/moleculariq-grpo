"""Report figures.

    python -m miqgrpo.plots training  --runs grpo-count-r001 grpo-index-r001 grpo-constraint-r001
    python -m miqgrpo.plots benchmark --results results/moleculariq
    python -m miqgrpo.plots all

Two sources, kept strictly apart:

* ``runs/<id>/logs/metrics.jsonl`` -- GRPO training reward and diagnostics.
  This is *training* signal on training-pool questions.
* ``results/moleculariq/<id>/`` -- official benchmark output.
  This is the held-out result, produced by the official harness.

Training reward and benchmark accuracy are never drawn on the same axes: they
are different quantities on different data, and putting them together would
invite exactly the confusion the project is set up to avoid.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .paths import BENCHMARK_RESULTS_ROOT, FIGURES_ROOT, RUNS_ROOT, ensure_dirs

__all__ = ["main"]

# --- palette (validated categorical slots 1-4, light surface) --------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8980"
GRID = "#e6e5e1"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

#: Diverging pair for "better/worse than baseline": blue <-> red, gray midpoint.
DIVERGING_LOW = "#2a78d6"
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#e34948"

#: Stable colour per model label, assigned by identity rather than by rank, so a
#: filtered or reordered chart never repaints the survivors.
MODEL_ORDER = ("baseline", "count", "index", "constraint_generation")
MODEL_LABELS = {
    "baseline": "Qwen2.5-0.5B-Instruct (base)",
    "count": "GRPO: count only",
    "index": "GRPO: index only",
    "constraint_generation": "GRPO: constrained generation only",
}
MODEL_COLORS = dict(zip(MODEL_ORDER, SERIES))

TASK_TYPES = ("count", "index", "generation")
TASK_LABELS = {
    "count": "Counting",
    "index": "Index attribution",
    "generation": "Constrained generation",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "font.size": 10,
        }
    )


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="x", visible=False)


def _save(fig, path: Path) -> Path:
    ensure_dirs(path.parent)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# training curves
# ---------------------------------------------------------------------------


def load_metrics(run_id: str) -> list[dict[str, Any]]:
    path = RUNS_ROOT / run_id / "logs" / "metrics.jsonl"
    if not path.exists():
        print(f"  ! no metrics for '{run_id}' at {path}")
        return []
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _series(records: Sequence[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    steps, values = [], []
    for record in records:
        if key in record and isinstance(record[key], (int, float)):
            steps.append(record.get("step", len(steps)))
            values.append(float(record[key]))
    return steps, values


#: Panels of the training figure. TRL names the aggregate reward "reward"; the
#: rest come from our own reward diagnostics via `log_metric`.
TRAINING_PANELS: tuple[tuple[str, str, str], ...] = (
    ("reward", "Total reward", "mean reward per completion"),
    ("reward/correctness_mean", "Verifier correctness", "fraction judged correct"),
    ("parse/answer_tag_fraction", "Answer-tag compliance", "fraction using <answer>"),
    ("completions/mean_length", "Completion length", "tokens"),
)


def plot_training(run_ids: Sequence[str], family_of: dict[str, str]) -> Path | None:
    _style()
    loaded = {run_id: load_metrics(run_id) for run_id in run_ids}
    loaded = {k: v for k, v in loaded.items() if v}
    if not loaded:
        print("  ! no training metrics found; skipping training figure")
        return None

    panels = [
        panel
        for panel in TRAINING_PANELS
        if any(_series(records, panel[0])[1] for records in loaded.values())
    ]
    if not panels:
        print("  ! metrics files contain none of the expected keys")
        return None

    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(4.6 * n, 3.4), squeeze=False, layout="constrained"
    )
    for ax, (key, title, ylabel) in zip(axes[0], panels):
        endpoints: list[tuple[float, str, str]] = []
        for run_id, records in loaded.items():
            family = family_of.get(run_id, run_id)
            color = MODEL_COLORS.get(family, INK_SECONDARY)
            steps, values = _series(records, key)
            if not values:
                continue
            # Raw series stays visible behind the trend; smoothing a noisy RL
            # curve without showing the noise would overstate how clean it is.
            ax.plot(steps, values, color=color, linewidth=1.0, alpha=0.25)
            smooth_steps, smooth_values = _rolling_mean(steps, values)
            ax.plot(
                smooth_steps,
                smooth_values,
                color=color,
                label=MODEL_LABELS.get(family, run_id),
                solid_capstyle="round",
            )
            endpoints.append(
                (smooth_values[-1], _short(family), color)
            )
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
        _clean(ax)
        _label_line_ends(ax, endpoints)

    handles = _legend_handles(list(loaded), family_of)
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(
        "GRPO training on MolecularIQ training-pool questions",
        fontsize=13,
        color=INK,
    )
    return _save(fig, FIGURES_ROOT / "training_curves.png")


#: Second training figure: the health checks, not the headline. Worth a panel in
#: a methods section, and the first place to look when a run misbehaves.
DIAGNOSTIC_PANELS: tuple[tuple[str, str, str], ...] = (
    ("reward_std", "Within-batch reward spread", "std of reward"),
    ("parse/well_formed_fraction", "Well-formed answers", "fraction"),
    ("parse/failure_fraction", "Extraction failures", "fraction"),
    ("grad_norm", "Gradient norm", "norm"),
)


def plot_training_diagnostics(
    run_ids: Sequence[str], family_of: dict[str, str]
) -> Path | None:
    _style()
    loaded = {run_id: load_metrics(run_id) for run_id in run_ids}
    loaded = {k: v for k, v in loaded.items() if v}
    panels = [
        panel
        for panel in DIAGNOSTIC_PANELS
        if any(_series(records, panel[0])[1] for records in loaded.values())
    ]
    if not panels:
        return None

    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.6 * len(panels), 3.4), squeeze=False, layout="constrained"
    )
    for ax, (key, title, ylabel) in zip(axes[0], panels):
        endpoints: list[tuple[float, str, str]] = []
        for run_id, records in loaded.items():
            family = family_of.get(run_id, run_id)
            color = MODEL_COLORS.get(family, INK_SECONDARY)
            steps, values = _series(records, key)
            if not values:
                continue
            ax.plot(steps, values, color=color, linewidth=1.0, alpha=0.25)
            smooth_steps, smooth_values = _rolling_mean(steps, values)
            ax.plot(smooth_steps, smooth_values, color=color, solid_capstyle="round")
            endpoints.append((smooth_values[-1], _short(family), color))
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
        _clean(ax)
        _label_line_ends(ax, endpoints)

    handles = _legend_handles(list(loaded), family_of)
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle("GRPO training diagnostics", fontsize=13, color=INK)
    return _save(fig, FIGURES_ROOT / "training_diagnostics.png")


def _legend_handles(run_ids: Sequence[str], family_of: dict[str, str]) -> list:
    """Legend entries for the runs actually plotted.

    Built from what was drawn rather than from MODEL_ORDER, so an ad-hoc run id
    (a smoke test, a re-run under a new name) still gets a legend instead of an
    empty one -- which matplotlib rejects outright.
    """
    seen: dict[str, str] = {}
    for run_id in run_ids:
        family = family_of.get(run_id, run_id)
        seen.setdefault(family, MODEL_LABELS.get(family, run_id))
    ordered = [f for f in MODEL_ORDER if f in seen] + [
        f for f in seen if f not in MODEL_ORDER
    ]
    return [
        plt.Line2D(
            [], [], color=MODEL_COLORS.get(f, INK_SECONDARY), lw=2, label=seen[f]
        )
        for f in ordered
    ]


def _short(family: str) -> str:
    return {"constraint_generation": "constraint"}.get(family, family)


def _rolling_mean(
    steps: Sequence[int], values: Sequence[float], window: int = 9
) -> tuple[list[int], list[float]]:
    if len(values) < window:
        return list(steps), list(values)
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return list(steps), smoothed


def _label_line_ends(ax, endpoints: Sequence[tuple[float, str, str]]) -> None:
    """Direct-label each line inside its own axes, nudged apart if they collide.

    Labels outside the axes overlap the next panel's tick labels; labels at
    their exact final value overlap each other whenever two runs converge. Both
    happen here, so place them inside and de-collide vertically.
    """
    if not endpoints:
        return
    low, high = ax.get_ylim()
    span = high - low or 1.0
    placed = sorted(
        ((value - low) / span, text, color) for value, text, color in endpoints
    )
    minimum_gap = 0.09
    for i in range(1, len(placed)):
        previous, current = placed[i - 1][0], placed[i][0]
        if current - previous < minimum_gap:
            placed[i] = (previous + minimum_gap, placed[i][1], placed[i][2])
    # Pushing apart upwards can run the stack off the top, where clamping would
    # pile the labels back on top of each other; slide the whole stack down.
    overflow = placed[-1][0] - 0.97
    if overflow > 0:
        placed = [(y - overflow, text, color) for y, text, color in placed]
    for fraction, text, color in placed:
        ax.annotate(
            text,
            xy=(0.985, min(0.97, max(0.03, fraction))),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=8,
            color=color,
        )


# ---------------------------------------------------------------------------
# benchmark results
# ---------------------------------------------------------------------------


def load_benchmark_runs(results_root: Path) -> dict[str, dict[str, Any]]:
    """Read every completed benchmark run, keyed by its model label."""
    runs: dict[str, dict[str, Any]] = {}
    for directory in sorted(results_root.glob("*")):
        manifest_path = directory / "eval_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("full_benchmark"):
            print(f"  skipping '{directory.name}' (not a whole-benchmark run)")
            continue
        label = manifest.get("label") or directory.name
        runs[label] = {
            "manifest": manifest,
            "directory": directory,
            "headline": manifest.get("headline") or {},
            "samples": load_samples(directory),
        }
    return runs


def load_samples(directory: Path) -> list[dict[str, Any]]:
    """Per-item records written by ``lm_eval --log_samples``.

    Used only to *break down* the official numbers by task type, complexity and
    multitask load. The headline metrics always come from the harness's own
    results file, never recomputed here.
    """
    samples: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("samples_*.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def _doc_field(record: dict[str, Any], field: str) -> Any:
    doc = record.get("doc") or {}
    return doc.get(field, record.get(field))


def breakdown(samples: Iterable[dict[str, Any]], field: str, metric: str) -> dict[str, float]:
    """Mean of one official per-item metric, grouped by a dataset field."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in samples:
        value = record.get(metric)
        if not isinstance(value, (int, float)):
            continue
        key = _doc_field(record, field)
        if key is None:
            continue
        buckets[str(key)].append(float(value))
    return {k: sum(v) / len(v) for k, v in sorted(buckets.items()) if v}


def plot_headline(runs: dict[str, dict[str, Any]]) -> Path | None:
    _style()
    labels = [label for label in MODEL_ORDER if label in runs]
    if not labels:
        print("  ! no benchmark runs found; skipping headline figure")
        return None

    metrics = ("pass_at_1", "pass_at_3", "avg_accuracy")
    metric_titles = {"pass_at_1": "pass@1", "pass_at_3": "pass@3", "avg_accuracy": "avg accuracy"}

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    group_width = 0.8
    bar_width = group_width / len(labels)
    for offset, label in enumerate(labels):
        headline = runs[label]["headline"]
        xs = [
            i - group_width / 2 + bar_width * (offset + 0.5) for i in range(len(metrics))
        ]
        ys = [float(headline.get(m) or 0.0) * 100 for m in metrics]
        bars = ax.bar(
            xs,
            ys,
            width=bar_width * 0.88,  # 2px-equivalent gap between adjacent bars
            color=MODEL_COLORS[label],
            label=MODEL_LABELS[label],
        )
        for bar, value in zip(bars, ys):
            # Direct labels: also the relief the contrast WARN requires.
            ax.annotate(
                f"{value:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK_SECONDARY,
            )

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([metric_titles[m] for m in metrics])
    ax.set_ylabel("score (%)")
    ax.set_title("Official MolecularIQ benchmark, whole test split (5,111 items)")
    # Below the axes rather than inside: which bar is tallest depends on the
    # data, so any in-axes corner can end up covered.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _clean(ax)
    return _save(fig, FIGURES_ROOT / "benchmark_headline.png")


def plot_by_task_type(runs: dict[str, dict[str, Any]], metric: str = "avg_accuracy") -> Path | None:
    _style()
    labels = [label for label in MODEL_ORDER if label in runs]
    if not labels:
        return None
    data = {label: breakdown(runs[label]["samples"], "task_type", metric) for label in labels}
    if not any(data.values()):
        print("  ! no per-sample logs; skipping task-type breakdown")
        return None

    present = [t for t in TASK_TYPES if any(t in d for d in data.values())]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    group_width = 0.8
    bar_width = group_width / len(labels)
    for offset, label in enumerate(labels):
        xs = [i - group_width / 2 + bar_width * (offset + 0.5) for i in range(len(present))]
        ys = [data[label].get(t, 0.0) * 100 for t in present]
        bars = ax.bar(
            xs, ys, width=bar_width * 0.88, color=MODEL_COLORS[label], label=MODEL_LABELS[label]
        )
        for bar, value in zip(bars, ys):
            ax.annotate(
                f"{value:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK_SECONDARY,
            )
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in present])
    ax.set_ylabel(f"{metric.replace('_', ' ')} (%)")
    ax.set_title("Benchmark accuracy by task type -- does single-task training transfer?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _clean(ax)
    return _save(fig, FIGURES_ROOT / "benchmark_by_task_type.png")


def plot_transfer_matrix(
    runs: dict[str, dict[str, Any]], metric: str = "avg_accuracy"
) -> Path | None:
    """Trained-on (rows) x evaluated task type (cols), as change vs baseline."""
    _style()
    if "baseline" not in runs:
        print("  ! no baseline run; skipping transfer matrix")
        return None
    trained = [label for label in MODEL_ORDER[1:] if label in runs]
    if not trained:
        return None

    base = breakdown(runs["baseline"]["samples"], "task_type", metric)
    present = [t for t in TASK_TYPES if t in base]
    if not present:
        print("  ! no per-sample logs; skipping transfer matrix")
        return None

    matrix = [
        [
            (breakdown(runs[label]["samples"], "task_type", metric).get(t, 0.0) - base[t])
            * 100
            for t in present
        ]
        for label in trained
    ]
    span = max(1.0, max(abs(v) for row in matrix for v in row))

    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list(
        "delta", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH]
    )
    # Blue = worse than baseline, red = better; gray means no change.
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

    fig, ax = plt.subplots(figsize=(1.9 * len(present) + 3.6, 1.0 * len(trained) + 2.6))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in present])
    ax.set_yticks(range(len(trained)))
    ax.set_yticklabels([f"trained on\n{t.replace('_', ' ')}" for t in trained])
    for row in range(len(trained)):
        for col in range(len(present)):
            value = matrix[row][col]
            ax.text(
                col,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=10,
                color=INK if abs(value) < span * 0.55 else SURFACE,
            )
    # 2px surface gap between adjacent fills, drawn as minor gridlines.
    ax.set_xticks([x - 0.5 for x in range(1, len(present))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(trained))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(
        f"Change in {metric.replace('_', ' ')} vs base model (percentage points)"
    )
    # Polarity is carried by the printed numbers; the caption keeps the colour
    # from being the only thing a reader has to decode.
    ax.text(
        0.0,
        -0.22,
        "red = above the base model   ·   gray = unchanged   ·   blue = below",
        transform=ax.transAxes,
        fontsize=8,
        color=INK_MUTED,
    )
    return _save(fig, FIGURES_ROOT / "transfer_matrix.png")


def plot_complexity(runs: dict[str, dict[str, Any]], metric: str = "avg_accuracy") -> Path | None:
    _style()
    labels = [label for label in MODEL_ORDER if label in runs]
    fields = [("complexity_bin", "Molecular complexity (Bertz)"), ("multi_task_load", "Multitask load")]
    panels = []
    for field, title in fields:
        data = {label: breakdown(runs[label]["samples"], field, metric) for label in labels}
        if any(data.values()):
            panels.append((field, title, data))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 3.8), squeeze=False)
    for ax, (field, title, data) in zip(axes[0], panels):
        keys = sorted({k for d in data.values() for k in d}, key=_sortable)
        group_width = 0.8
        bar_width = group_width / max(1, len(labels))
        for offset, label in enumerate(labels):
            xs = [i - group_width / 2 + bar_width * (offset + 0.5) for i in range(len(keys))]
            ys = [data[label].get(k, 0.0) * 100 for k in keys]
            ax.bar(
                xs, ys, width=bar_width * 0.88, color=MODEL_COLORS[label], label=MODEL_LABELS[label]
            )
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys)
        ax.set_ylabel(f"{metric.replace('_', ' ')} (%)")
        ax.set_title(title)
        _clean(ax)
    handles = [Patch(color=MODEL_COLORS[l], label=MODEL_LABELS[l]) for l in labels]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    return _save(fig, FIGURES_ROOT / "benchmark_by_complexity.png")


def _sortable(key: str) -> tuple[int, str]:
    try:
        return (0, f"{float(key):012.3f}")
    except ValueError:
        return (1, key)


def write_table(runs: dict[str, dict[str, Any]]) -> Path:
    """The table view that the palette's contrast WARN obliges us to ship."""
    rows = []
    for label in MODEL_ORDER:
        if label not in runs:
            continue
        entry = runs[label]
        row = {
            "model": MODEL_LABELS[label],
            "evaluation_run_id": entry["manifest"].get("evaluation_run_id"),
            "checkpoint": entry["manifest"].get("checkpoint"),
            "full_benchmark": entry["manifest"].get("full_benchmark"),
        }
        for metric, value in (entry["headline"] or {}).items():
            row[metric] = value
        for task, value in breakdown(entry["samples"], "task_type", "avg_accuracy").items():
            row[f"avg_accuracy::{task}"] = round(value, 4)
        rows.append(row)

    path = FIGURES_ROOT / "benchmark_summary.csv"
    ensure_dirs(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_RUNS = {
    "grpo-count-r001": "count",
    "grpo-index-r001": "index",
    "grpo-constraint-r001": "constraint_generation",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    training = sub.add_parser("training", help="GRPO training curves")
    training.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))

    bench = sub.add_parser("benchmark", help="official benchmark figures")
    bench.add_argument("--results", type=Path, default=BENCHMARK_RESULTS_ROOT)
    bench.add_argument("--metric", default="avg_accuracy")

    every = sub.add_parser("all", help="every figure")
    every.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    every.add_argument("--results", type=Path, default=BENCHMARK_RESULTS_ROOT)
    every.add_argument("--metric", default="avg_accuracy")

    args = parser.parse_args(argv)

    if args.command in ("training", "all"):
        print("training figures:")
        family_of = {
            run_id: DEFAULT_RUNS.get(run_id, run_id) for run_id in args.runs
        }
        plot_training(args.runs, family_of)
        plot_training_diagnostics(args.runs, family_of)

    if args.command in ("benchmark", "all"):
        print("benchmark figures:")
        runs = load_benchmark_runs(args.results)
        if not runs:
            print(f"  ! nothing in {args.results}")
            return
        plot_headline(runs)
        plot_by_task_type(runs, args.metric)
        plot_transfer_matrix(runs, args.metric)
        plot_complexity(runs, args.metric)
        write_table(runs)


if __name__ == "__main__":
    main()
