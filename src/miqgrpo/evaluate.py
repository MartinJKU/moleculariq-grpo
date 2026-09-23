"""Official MolecularIQ benchmark evaluation.

    python -m miqgrpo.evaluate run \
        --run-id miq-eval-baseline \
        --model-path Qwen/Qwen2.5-0.5B-Instruct \
        --label baseline

This shells out to the official ``moleculariq-eval`` harness (a fork of
lm-evaluation-harness with the MolecularIQ task built in) and runs the **whole**
benchmark: all 5,111 test items, the official ``moleculariq_pass_at_k`` task
config, the official system instruction, the official extraction and the
official verifier. Nothing here re-implements scoring.

The benchmark is test-only. ``--limit`` is refused unless ``--smoke`` is passed,
and a smoke run is written to a separate directory with ``full_benchmark:
false`` stamped in its manifest so it can never be quoted as a result.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .paths import ensure_dirs, evaluation_dir, refuse_to_overwrite, run_dir
from .prompts import SYSTEM_PROMPT
from .provenance import capture, environment_report, sha256_tree, write_json

__all__ = ["run_benchmark", "main"]

#: The official task. `moleculariq_inline` exists for non-chat models; Qwen2.5
#: Instruct is a chat model, so the system-instruction variant is the right one.
OFFICIAL_TASK = "moleculariq_pass_at_k"

#: Metrics the official task emits and that we report verbatim.
OFFICIAL_METRICS = ("pass_at_1", "pass_at_3", "avg_accuracy")


def _require_lm_eval(optional: bool = False) -> str:
    """Locate the official harness. ``optional`` keeps --dry-run usable."""
    executable = shutil.which("lm_eval")
    if executable is not None:
        return executable
    if optional:
        return "lm_eval"
    raise SystemExit(
        "lm_eval not found on PATH.\n"
        "Install the official harness:\n"
        "  git clone https://github.com/ml-jku/moleculariq-eval.git\n"
        "  pip install -e 'moleculariq-eval[vllm]'\n"
        "  pip install moleculariq-core rdkit"
    )


def _harness_version() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import lm_eval

        info["lm_eval_version"] = getattr(lm_eval, "__version__", None)
        info["lm_eval_path"] = str(Path(lm_eval.__file__).parent)
        repo = Path(lm_eval.__file__).parent.parent
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        info["moleculariq_eval_commit"] = (
            commit.stdout.strip() if commit.returncode == 0 else None
        )
    except Exception as exc:  # noqa: BLE001
        info["lm_eval_import_error"] = str(exc)
    return info


def _training_provenance(experiment_id: str | None) -> dict[str, Any]:
    """Pull the training side of the chain so the result is self-describing."""
    if not experiment_id:
        return {}
    directory = run_dir(experiment_id)
    provenance: dict[str, Any] = {"experiment_id": experiment_id}
    for name, key in (
        ("provenance.json", "training_provenance"),
        ("training_summary.json", "training_summary"),
    ):
        path = directory / name
        if path.exists():
            provenance[key] = json.loads(path.read_text())
    config_path = directory / "frozen_config.yaml"
    if config_path.exists():
        provenance["frozen_config"] = yaml.safe_load(config_path.read_text())
    return provenance


def build_command(
    model_path: str,
    output_path: Path,
    backend: str,
    batch_size: str,
    dtype: str,
    gpu_memory_utilization: float,
    limit: int | None,
    gen_kwargs: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> list[str]:
    if backend == "vllm":
        model_args = (
            f"pretrained={model_path},dtype={dtype},"
            f"gpu_memory_utilization={gpu_memory_utilization}"
        )
    else:
        model_args = f"pretrained={model_path},dtype={dtype}"

    command = [
        _require_lm_eval(optional=dry_run),
        "--model",
        backend,
        "--model_args",
        model_args,
        "--tasks",
        OFFICIAL_TASK,
        "--apply_chat_template",
        "--system_instruction",
        SYSTEM_PROMPT,
        "--batch_size",
        batch_size,
        "--log_samples",
        "--output_path",
        str(output_path),
    ]
    if gen_kwargs:
        # Two things force explicit generation kwargs on the HF backend, and
        # both are about *matching* the official vLLM runs rather than departing
        # from them -- see docs/official-semantics.md.
        #
        #   max_gen_toks: the task YAML asks for 32768 new tokens, exactly this
        #     model's context window, so max_ctx_len = max_length - max_gen_toks
        #     is 0 and the run asserts. vLLM survives the same arithmetic only
        #     through a tokens[-0:] quirk that returns the whole list.
        #
        #   temperature/top_p/top_k/repetition_penalty: the task sets
        #     do_sample=true and no temperature. The vLLM backend drops
        #     do_sample and leaves temperature unset, so vLLM's SamplingParams
        #     defaults apply (1.0 / 1.0 / off / 1.0). The HF backend instead
        #     injects temperature=0.0 and then crashes on do_sample=true; and if
        #     it did not, HF would fall back to Qwen's own generation_config
        #     (0.7 / 0.8 / 20 / 1.1), which is not what the official runs used.
        #
        # Recorded in the manifest under `generation_overrides`.
        rendered = ",".join(f"{key}={value}" for key, value in gen_kwargs.items())
        command += ["--gen_kwargs", rendered]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def _collect_results(raw_dir: Path) -> dict[str, Any]:
    """Find the harness's own result JSON; do not recompute anything from it."""
    candidates = sorted(raw_dir.rglob("results_*.json"))
    if not candidates:
        return {}
    payload = json.loads(candidates[-1].read_text())
    task_results = (payload.get("results") or {}).get(OFFICIAL_TASK, {})
    headline = {
        metric: task_results.get(f"{metric},all", task_results.get(metric))
        for metric in OFFICIAL_METRICS
    }
    return {
        "results_file": str(candidates[-1]),
        "task_results": task_results,
        "headline": headline,
        "n_samples": payload.get("n-samples"),
        "config": payload.get("config"),
        "versions": payload.get("versions"),
    }


def run_benchmark(
    run_id: str,
    model_path: str,
    label: str,
    experiment_id: str | None = None,
    backend: str = "vllm",
    batch_size: str = "auto",
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.85,
    limit: int | None = None,
    gen_kwargs: dict[str, Any] | None = None,
    smoke: bool = False,
    dry_run: bool = False,
) -> Path:
    if limit is not None and not smoke:
        raise SystemExit(
            "--limit truncates the benchmark, so the result would not be a "
            "MolecularIQ score. Re-run with --smoke to mark it as an "
            "infrastructure test instead."
        )
    if smoke and limit is None:
        limit = 10

    out_dir = evaluation_dir(run_id if not smoke else f"{run_id}-smoke")
    refuse_to_overwrite(out_dir, f"benchmark result '{run_id}'")
    raw_dir = out_dir / "raw"
    ensure_dirs(out_dir, raw_dir)

    command = build_command(
        model_path,
        raw_dir,
        backend,
        batch_size,
        dtype,
        gpu_memory_utilization,
        limit,
        gen_kwargs=gen_kwargs,
        dry_run=dry_run,
    )

    checkpoint_hash = None
    local_checkpoint = Path(model_path)
    if local_checkpoint.exists() and local_checkpoint.is_dir():
        checkpoint_hash = sha256_tree(local_checkpoint)

    manifest: dict[str, Any] = {
        "evaluation_run_id": run_id,
        "label": label,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": model_path,
        "checkpoint_hash": checkpoint_hash,
        "task": OFFICIAL_TASK,
        "backend": backend,
        "batch_size": batch_size,
        "dtype": dtype,
        "limit": limit,
        "full_benchmark": limit is None,
        "system_instruction": SYSTEM_PROMPT,
        "system_instruction_source": "moleculariq-eval task_processor.SYSTEM_PROMPT",
        "apply_chat_template": True,
        # Repeats, sampling temperature, stop rules and answer extraction all
        # come from the official task YAML. The only override is the generation
        # cap, and only because the published value is unrunnable on a model
        # whose context equals it -- see build_command.
        "generation_overrides": dict(gen_kwargs) if gen_kwargs else None,
        "command": command,
        "harness": _harness_version(),
        "training": _training_provenance(experiment_id),
        "integrity": {
            "benchmark_used_for_training": False,
            "benchmark_used_for_hparam_selection": False,
            "benchmark_used_for_prompt_tuning": False,
            "benchmark_used_for_extraction_tuning": False,
            "benchmark_used_for_checkpoint_selection": False,
            "is_infrastructure_smoke_test": bool(smoke),
        },
    }

    print(f"evaluation run : {run_id}")
    print(f"model          : {model_path}")
    print(f"task           : {OFFICIAL_TASK} (whole benchmark: {limit is None})")
    print(f"output         : {out_dir}")
    print()
    print("command:")
    print("  " + " ".join(_quote(part) for part in command))
    print()

    if dry_run:
        write_json(out_dir / "eval_manifest.json", manifest)
        print("dry run; nothing executed")
        return out_dir

    (out_dir / "environment.txt").write_text(
        json.dumps(environment_report(), indent=2, default=str) + "\n"
    )

    with open(out_dir / "stdout.log", "w") as stdout, open(
        out_dir / "stderr.log", "w"
    ) as stderr:
        completed = subprocess.run(
            command, stdout=stdout, stderr=stderr, check=False, env=os.environ.copy()
        )

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["returncode"] = completed.returncode
    manifest.update(_collect_results(raw_dir))
    write_json(out_dir / "eval_manifest.json", manifest)
    with open(out_dir / "eval_manifest.yaml", "w") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    write_json(out_dir / "provenance.json", capture("evaluate", {"run_id": run_id}))

    if completed.returncode != 0:
        print(f"lm_eval exited {completed.returncode}; see {out_dir/'stderr.log'}")
        sys.exit(completed.returncode)

    headline = manifest.get("headline") or {}
    print("official metrics:")
    for metric, value in headline.items():
        print(f"  {metric:14s} {value}")
    summary = {
        "evaluation_run_id": run_id,
        "label": label,
        "checkpoint": model_path,
        "full_benchmark": manifest["full_benchmark"],
        "metrics": headline,
        "task_results": manifest.get("task_results", {}),
    }
    write_json(out_dir / "summary.json", summary)
    return out_dir


def parse_gen_kwargs(raw: str | None) -> dict[str, Any] | None:
    """Parse ``k=v,k=v`` into an ordered mapping, preserving written form."""
    if not raw:
        return None
    parsed: dict[str, Any] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"malformed --gen-kwargs entry {item!r}; expected key=value")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed or None


def _quote(part: str) -> str:
    return f"'{part}'" if any(c in part for c in " \n\"'") else part


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the official benchmark on one model")
    run.add_argument("--run-id", required=True, help="unique evaluation_run_id")
    run.add_argument("--model-path", required=True, help="HF model id or checkpoint dir")
    run.add_argument("--label", required=True, help="short name used in plots")
    run.add_argument("--experiment", default=None, help="training experiment_id, if any")
    run.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    run.add_argument("--batch-size", default="auto")
    run.add_argument("--dtype", default="bfloat16")
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run.add_argument(
        "--gen-kwargs",
        default=None,
        help=(
            "comma-separated key=value generation overrides passed through to "
            "lm_eval, e.g. 'max_gen_toks=28672,temperature=1.0'. Recorded in the "
            "run manifest under generation_overrides"
        ),
    )
    run.add_argument("--limit", type=int, default=None)
    run.add_argument(
        "--smoke",
        action="store_true",
        help="infrastructure test on a handful of items; never a reportable score",
    )
    run.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    run_benchmark(
        run_id=args.run_id,
        model_path=args.model_path,
        label=args.label,
        experiment_id=args.experiment,
        backend=args.backend,
        batch_size=args.batch_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit=args.limit,
        gen_kwargs=parse_gen_kwargs(args.gen_kwargs),
        smoke=args.smoke,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
