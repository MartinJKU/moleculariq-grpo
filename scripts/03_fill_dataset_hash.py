#!/usr/bin/env python3
"""Pin each experiment config to the dataset artifact it was built against.

    python scripts/03_fill_dataset_hash.py --artifact miq-train-v001

Writes the artifact's content hash into every matching experiment config, so a
training run refuses to start if the data changed underneath a fixed experiment
ID. Without this the "frozen dataset" is only frozen by convention.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from miqgrpo.paths import REPO_ROOT, dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--configs", type=Path, default=REPO_ROOT / "configs" / "experiments")
    args = parser.parse_args()

    manifest = json.loads((dataset_dir(args.artifact) / "manifest.json").read_text())
    digest = manifest["dataset_hash"]
    print(f"{args.artifact} -> {digest}")

    for path in sorted(args.configs.glob("*.yaml")):
        text = path.read_text()
        if f"artifact_id: {args.artifact}" not in text:
            continue
        updated = re.sub(
            r"^(\s*)expected_hash:.*$",
            rf"\1expected_hash: {digest}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if updated != text:
            path.write_text(updated)
            print(f"  pinned {path.name}")


if __name__ == "__main__":
    main()
