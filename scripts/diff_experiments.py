#!/usr/bin/env python3
"""Diff two resolved experiment configs.

    python scripts/diff_experiments.py grpo-index-r001 grpo-index-r002

Before attributing an outcome difference to one variable, check that one
variable is all that changed. It is easy to copy a config, change the thing you
meant to change, and also pick up a different dataset artifact, seed or
generation setting without noticing -- at which point the comparison says
nothing.

Reads runs/<id>/frozen_config.yaml when the run exists (what actually executed),
otherwise configs/experiments/<id>.yaml (what is intended to).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from miqgrpo.config import load_experiment_config
from miqgrpo.paths import REPO_ROOT, run_dir

#: Fields whose difference is expected and carries no behavioural meaning.
COSMETIC = {"id", "notes"}


def resolved(experiment_id: str) -> tuple[dict[str, Any], str]:
    frozen = run_dir(experiment_id) / "frozen_config.yaml"
    if frozen.exists():
        return yaml.safe_load(frozen.read_text()), f"{frozen} (as executed)"
    config_path = REPO_ROOT / "configs" / "experiments" / f"{experiment_id}.yaml"
    if not config_path.exists():
        raise SystemExit(f"no config or run found for '{experiment_id}'")
    return load_experiment_config(config_path).as_dict(), f"{config_path} (not yet run)"


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out.update(flatten(value, f"{prefix}{key}."))
    else:
        out[prefix.rstrip(".")] = data
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()

    left, left_src = resolved(args.left)
    right, right_src = resolved(args.right)
    print(f"left : {args.left}\n       {left_src}")
    print(f"right: {args.right}\n       {right_src}\n")

    flat_left, flat_right = flatten(left), flatten(right)
    keys = sorted(set(flat_left) | set(flat_right))

    behavioural: list[str] = []
    cosmetic: list[str] = []
    for key in keys:
        a, b = flat_left.get(key, "<absent>"), flat_right.get(key, "<absent>")
        if a == b:
            continue
        line = f"  {key}\n      {a!r}\n   -> {b!r}"
        (cosmetic if key.split(".")[-1] in COSMETIC else behavioural).append(line)

    if behavioural:
        print(f"behaviour-changing differences ({len(behavioural)}):")
        print("\n".join(behavioural))
    else:
        print("no behaviour-changing differences")

    if cosmetic:
        print(f"\ncosmetic ({len(cosmetic)}): {', '.join(c.split()[0] for c in cosmetic)}")

    print()
    if len(behavioural) == 1:
        print("Exactly one variable differs -- a controlled comparison.")
    elif len(behavioural) > 1:
        print(
            f"{len(behavioural)} variables differ. An outcome difference cannot be\n"
            f"attributed to any single one of them."
        )


if __name__ == "__main__":
    main()
