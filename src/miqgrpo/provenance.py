"""Provenance capture.

A number in the report is only worth something if we can say which code,
environment, dataset artifact and checkpoint produced it. Every stage writes a
``provenance.json`` next to its outputs using :func:`capture`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT

__all__ = [
    "capture",
    "environment_report",
    "git_state",
    "sha256_file",
    "sha256_tree",
    "hash_config",
    "write_json",
]

#: Packages whose version changes can change results.
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "accelerate",
    "datasets",
    "peft",
    "rdkit",
    "moleculariq_core",
    "numpy",
    "vllm",
    "lm_eval",
)


def _run_git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    """Commit, dirty flag and the diff itself when the tree is dirty.

    A dirty tree is recorded rather than rejected -- but the diff is stored so a
    result can still be reproduced exactly.
    """
    commit = _run_git("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    status = _run_git("status", "--porcelain") or ""
    dirty = bool(status.strip())
    state: dict[str, Any] = {
        "available": True,
        "commit": commit,
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
    }
    if dirty:
        state["status"] = status
        state["diff"] = _run_git("diff", "HEAD") or ""
    return state


def _package_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        dist_name = name.replace("_", "-")
        try:
            versions[name] = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
    return versions


def _gpu_report() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_available": False}

    report: dict[str, Any] = {
        "torch_available": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if report["cuda_available"]:
        report["devices"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1024**3, 2
                ),
                "capability": ".".join(
                    str(x) for x in torch.cuda.get_device_capability(i)
                ),
            }
            for i in range(torch.cuda.device_count())
        ]
    return report


def environment_report() -> dict[str, Any]:
    """Everything about the machine that can move a number."""
    slurm = {
        key: value for key, value in os.environ.items() if key.startswith("SLURM_")
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": _package_versions(),
        "gpu": _gpu_report(),
        "slurm": slurm or None,
        "env": {
            key: os.environ.get(key)
            for key in (
                "HF_HOME",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "CUDA_VISIBLE_DEVICES",
                "MIQ_DATA",
                "MIQ_RUNS",
                "MIQ_RESULTS",
            )
        },
    }


def capture(stage: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one provenance record for a pipeline stage."""
    record: dict[str, Any] = {
        "stage": stage,
        "git": git_state(),
        "environment": environment_report(),
    }
    if extra:
        record.update(extra)
    return record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Order-independent content hash of a directory.

    Used to pin a dataset artifact or a checkpoint so that a training run and a
    benchmark run can prove they used the same bytes.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def hash_config(config: Any) -> str:
    """Stable hash of a config object (dicts are key-sorted)."""
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
