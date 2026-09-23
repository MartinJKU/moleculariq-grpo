"""Build the frozen MolecularIQ GRPO training dataset.

Run once, on CPU, before any GPU time is spent:

    python -m miqgrpo.build_dataset build --config configs/preprocessing/miq-train-v001.yaml
    python -m miqgrpo.build_dataset verify --artifact miq-train-v001

The output is an immutable artifact -- a ``datasets`` ``DatasetDict`` plus a
manifest that pins the pool revision, the ``moleculariq-core`` commit, the
config hash and a content hash of the saved bytes. Training loads it and nothing
else; every question, target and constraint already exists before the first
rollout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml
from datasets import Dataset, DatasetDict, load_from_disk

from . import __version__
from .generation import (
    TASK_FAMILIES,
    GeneratedExample,
    GenerationSpec,
    PropertyEngine,
    QuestionGenerator,
    ReferenceTable,
    build_property_catalog,
    oracle_score,
    recompute_target,
)
from .paths import dataset_dir, ensure_dirs, refuse_to_overwrite
from .prompts import build_prompt_messages
from .provenance import capture, hash_config, sha256_tree, write_json

#: Dataset identifiers that must never be used as preprocessing input.
FORBIDDEN_SOURCES = (
    "moleculariq-v0.0",
    "ml-jku/moleculariq-v0.0",
    "moleculariq_benchmark",
    "val_easy",
    "val_hard",
    "test",
)

TRAIN_POOL_ID = "ml-jku/moleculariq-trainPool"

#: Columns the dataset carries. Model-visible data is `prompt`/`question`;
#: everything else is reward-side or provenance and never enters the prompt.
DATASET_COLUMNS = (
    "example_id",
    "prompt",
    "question",
    "task_family",
    "task_type",
    "target_json",
    "constraints_json",
    "witness_smiles",
    "question_smiles",
    "molecule_id",
    "properties_json",
    "feature",
    "supercategory",
    "molecular_complexity",
    "complexity_bin",
    "multitask_load",
    "is_randomized",
    "is_kekulized",
    "constraint_prevalence",
    "question_seed",
    "dataset_artifact_id",
)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        config = yaml.safe_load(fh)
    _check_sources(config)
    return config


def _check_sources(config: dict[str, Any]) -> None:
    """Fail closed if anything benchmark-shaped is named as an input."""
    source = config.get("source", {})
    pool = str(source.get("dataset_id", TRAIN_POOL_ID))
    split = str(source.get("split", "train"))
    blob = f"{pool}::{split}".lower()
    for forbidden in FORBIDDEN_SOURCES:
        if forbidden.lower() in blob:
            raise ValueError(
                f"preprocessing input '{blob}' looks like the official benchmark "
                f"or a hidden pool ('{forbidden}'). Training data may only come "
                f"from the MolecularIQ training pool."
            )
    if pool != TRAIN_POOL_ID:
        raise ValueError(
            f"unexpected molecule source '{pool}'; expected '{TRAIN_POOL_ID}'"
        )


# ---------------------------------------------------------------------------
# molecule pool
# ---------------------------------------------------------------------------


def load_training_molecules(
    dataset_id: str,
    split: str,
    cache_dir: str | None = None,
    local_parquet: str | None = None,
) -> tuple[list[str], str | None]:
    """Load the official training pool and resolve its revision.

    Uses ``datasets`` directly rather than ``moleculariq_core.load_molecule_pool``
    so that the resolved commit hash can be recorded in the manifest -- the pool
    is the root of the provenance chain, and "whatever was on the Hub that day"
    is not good enough for a reportable run.

    ``local_parquet`` loads a pre-staged copy instead, for machines with no
    outbound network during preprocessing.
    """
    from datasets import load_dataset

    revision = _resolve_revision(dataset_id)
    if local_parquet:
        dataset = load_dataset("parquet", data_files=local_parquet, split="train")
    else:
        dataset = load_dataset(
            dataset_id, split=split, cache_dir=cache_dir, revision=revision
        )
    column = "smiles" if "smiles" in dataset.column_names else dataset.column_names[0]
    return list(dataset[column]), revision


def _resolve_revision(dataset_id: str) -> str | None:
    """Pin the pool to a commit SHA when the Hub is reachable."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(dataset_id).sha
    except Exception:
        return None


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _leaks_target(question: str, target: dict[str, Any] | None) -> str | None:
    """Detect a target accidentally serialised into the model-visible prompt.

    Count/index questions name the JSON *keys* on purpose; what must never
    appear is a key bound to its answer.
    """
    if not target:
        return None
    haystack = question.replace(" ", "")
    if json.dumps(target).replace(" ", "") in haystack:
        return "full target JSON in question"
    for key, value in target.items():
        for pattern in (f'"{key}":{json.dumps(value)}', f"{key}:{json.dumps(value)}"):
            if pattern.replace(" ", "") in haystack:
                return f"key/value pair '{key}' in question"
    return None


def validate_example(
    example: GeneratedExample, engine: PropertyEngine | None = None
) -> str | None:
    """Return a drop reason, or None when the example is sound.

    With ``engine`` supplied the stored target is also recomputed from the
    displayed SMILES, which is the check that catches a target that does not
    actually describe the molecule in the question.
    """
    if not example.question or not example.question.strip():
        return "empty_question"
    if example.task_family in ("count", "index"):
        if not example.target:
            return "missing_target"
        for value in example.target.values():
            if isinstance(value, float) and value != value:
                return "non_finite_target"
        leak = _leaks_target(example.question, example.target)
        if leak:
            return f"prompt_target_leakage:{leak}"
        if example.question_smiles and example.question_smiles not in example.question:
            return "question_smiles_missing_from_question"
        if engine is not None:
            recomputed = recompute_target(
                example.question_smiles, example.properties, example.task_family, engine
            )
            if recomputed != example.target:
                return "target_does_not_match_molecule"
    elif example.task_family == "constraint_generation":
        if not example.constraints:
            return "missing_constraints"
    else:
        return "unknown_task_family"
    if oracle_score(example) != 1.0:
        return "oracle_rejected"
    return None


# ---------------------------------------------------------------------------
# record assembly
# ---------------------------------------------------------------------------


def to_record(
    example: GeneratedExample, example_id: str, artifact_id: str
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "prompt": build_prompt_messages(example.question),
        "question": example.question,
        "task_family": example.task_family,
        "task_type": example.task_type,
        "target_json": json.dumps(example.target) if example.target is not None else None,
        "constraints_json": (
            json.dumps(example.constraints) if example.constraints is not None else None
        ),
        "witness_smiles": example.witness_smiles,
        "question_smiles": example.question_smiles,
        "molecule_id": example.molecule_id,
        "properties_json": json.dumps(example.properties),
        "feature": example.feature,
        "supercategory": example.supercategory,
        "molecular_complexity": float(example.molecular_complexity),
        "complexity_bin": example.complexity_bin,
        "multitask_load": int(example.multitask_load),
        "is_randomized": bool(example.is_randomized),
        "is_kekulized": bool(example.is_kekulized),
        "constraint_prevalence": (
            float(example.constraint_prevalence)
            if example.constraint_prevalence is not None
            else None
        ),
        "question_seed": int(example.question_seed),
        "dataset_artifact_id": artifact_id,
    }


def _distribution(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record[key]) for record in records))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build(config_path: Path, overwrite: bool = False) -> Path:
    config = load_config(config_path)
    artifact_id = config["dataset_artifact_id"]
    out_dir = dataset_dir(artifact_id)
    if not overwrite:
        refuse_to_overwrite(out_dir, f"dataset artifact '{artifact_id}'")
    ensure_dirs(out_dir)

    seed = int(config["generation"]["seed"])
    source = config.get("source", {})
    print(f"[1/7] loading training pool {source.get('dataset_id', TRAIN_POOL_ID)}")
    molecules, pool_revision = load_training_molecules(
        source.get("dataset_id", TRAIN_POOL_ID),
        source.get("split", "train"),
        source.get("cache_dir"),
        source.get("local_parquet"),
    )
    print(f"      {len(molecules)} molecules in pool")

    import random

    rng = random.Random(seed)
    n_reference = int(config["generation"].get("n_reference_molecules", 20000))
    n_reference = min(n_reference, len(molecules))
    reference_molecules = rng.sample(molecules, n_reference)

    print("[2/7] probing property support against the official verifier")
    engine = PropertyEngine(seed=seed)
    catalog = build_property_catalog(engine)
    summary = catalog.summary()
    print(
        f"      usable: {summary['n_count_properties']} count / "
        f"{summary['n_index_properties']} index / "
        f"{summary['n_constraint_properties']} constraint "
        f"({summary['n_rejected']} rejected)"
    )

    n_workers = int(config["generation"].get("n_workers", 0))
    print(
        f"[3/7] building reference statistics over {n_reference} molecules"
        + (f" ({n_workers} workers)" if n_workers > 1 else "")
    )
    table = ReferenceTable.build(
        reference_molecules, catalog, engine, progress=True, n_workers=n_workers
    )
    print(f"      {len(table)} molecules usable")

    print("[4/7] generating questions")
    splits: dict[str, list[dict[str, Any]]] = {}
    drop_counts: dict[str, dict[str, int]] = {}
    generator_stats: dict[str, Any] = {}

    for family in TASK_FAMILIES:
        family_config = config["generation"]["tasks"].get(family)
        if not family_config or not family_config.get("n_examples"):
            continue
        spec = GenerationSpec(
            family=family,
            n_examples=int(family_config["n_examples"]),
            multitask_weights={
                int(k): float(v)
                for k, v in (family_config.get("multitask_weights") or {1: 1.0}).items()
            },
            complexity_weights={
                str(k): float(v)
                for k, v in (
                    family_config.get("complexity_weights")
                    or config["generation"].get("complexity_weights")
                    or {"0-250": 0.5, "250-1000": 0.35, "1000-inf": 0.15}
                ).items()
            },
            randomize_prob=float(
                family_config.get("randomize_prob", config["generation"].get("randomize_prob", 0.5))
            ),
            kekulize_prob=float(
                family_config.get("kekulize_prob", config["generation"].get("kekulize_prob", 0.5))
            ),
            max_constraint_prevalence=float(
                family_config.get("max_constraint_prevalence", 0.03)
            ),
            min_constraint_prevalence=float(
                family_config.get("min_constraint_prevalence", 0.0)
            ),
        )
        generator = QuestionGenerator(
            seed=seed + abs(hash(family)) % 10_000, catalog=catalog, table=table, engine=engine
        )

        records: list[dict[str, Any]] = []
        drops: Counter = Counter()
        seen_ids: set[str] = set()
        for index, example in enumerate(generator.generate(spec)):
            reason = validate_example(example, engine)
            if reason:
                drops[reason] += 1
                continue
            example_id = f"{artifact_id}-{family}-{index:07d}"
            if example_id in seen_ids:
                drops["duplicate_example_id"] += 1
                continue
            seen_ids.add(example_id)
            records.append(to_record(example, example_id, artifact_id))

        drops.update(generator.drops)
        drop_counts[family] = dict(drops)
        generator_stats[family] = {
            "property_usage": dict(generator.property_usage),
            "category_usage": dict(generator.category_usage),
        }

        dev_fraction = float(config["generation"].get("dev_fraction", 0.05))
        split_rng = random.Random(seed + 1)
        split_rng.shuffle(records)
        n_dev = max(1, int(len(records) * dev_fraction)) if dev_fraction else 0
        splits[f"{family}_dev"] = records[:n_dev]
        splits[family] = records[n_dev:]
        print(
            f"      {family}: {len(splits[family])} train + {len(splits[f'{family}_dev'])} dev"
            f"  (dropped {sum(drops.values())})"
        )

    print("[5/7] materialising DatasetDict")
    dataset = DatasetDict(
        {name: Dataset.from_list(rows) for name, rows in splits.items() if rows}
    )
    for name, split in dataset.items():
        missing = set(DATASET_COLUMNS) - set(split.column_names)
        if missing:
            raise ValueError(f"split '{name}' is missing columns: {sorted(missing)}")
    dataset_path = out_dir / "dataset"
    dataset.save_to_disk(str(dataset_path))

    print("[6/7] writing manifest")
    config_hash = hash_config(config)
    manifest = {
        "dataset_artifact_id": artifact_id,
        "builder_version": __version__,
        "source_dataset": source.get("dataset_id", TRAIN_POOL_ID),
        "source_split": source.get("split", "train"),
        "source_dataset_revision": pool_revision,
        "source_pool_size": len(molecules),
        "n_reference_molecules": n_reference,
        "n_reference_molecules_usable": len(table),
        "random_seed": seed,
        "preprocessing_config": config,
        "preprocessing_config_hash": config_hash,
        "property_catalog": summary,
        "counts_by_split": {name: len(split) for name, split in dataset.items()},
        "num_examples": sum(len(split) for split in dataset.values()),
        "num_unique_molecules": len(
            {
                record["molecule_id"]
                for rows in splits.values()
                for record in rows
            }
        ),
        "drop_counts_by_reason": drop_counts,
        "generator_stats": generator_stats,
        "distributions": {
            name: {
                "feature": _distribution(splits[name], "feature"),
                "supercategory": _distribution(splits[name], "supercategory"),
                "complexity_bin": _distribution(splits[name], "complexity_bin"),
                "multitask_load": _distribution(splits[name], "multitask_load"),
            }
            for name in splits
            if splits[name]
        },
        "integrity": {
            "official_benchmark_used_as_input": False,
            "hidden_pools_used": False,
            "questions_generated_offline": True,
            "forbidden_sources_checked": list(FORBIDDEN_SOURCES),
        },
    }
    manifest["dataset_hash"] = sha256_tree(dataset_path)
    write_json(out_dir / "manifest.json", manifest)
    with open(out_dir / "preprocessing_config.yaml", "w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    write_json(out_dir / "provenance.json", capture("build_dataset", {"artifact_id": artifact_id}))

    print("[7/7] done")
    print(f"      artifact : {out_dir}")
    print(f"      hash     : {manifest['dataset_hash']}")
    print(f"      examples : {manifest['num_examples']}")
    print()
    print("Now verify it in a fresh process:")
    print(f"      python -m miqgrpo.build_dataset verify --artifact {artifact_id}")
    return out_dir


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def verify(artifact_id: str) -> None:
    """Reload the artifact from scratch and re-check every invariant.

    Run as a separate process: a dataset that only works inside the builder's
    memory is not a frozen artifact.
    """
    out_dir = dataset_dir(artifact_id)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    dataset_path = out_dir / "dataset"

    dataset = load_from_disk(str(dataset_path))
    engine = PropertyEngine()
    failures: list[str] = []

    actual_hash = sha256_tree(dataset_path)
    if actual_hash != manifest["dataset_hash"]:
        failures.append(
            f"content hash changed: manifest {manifest['dataset_hash']} != {actual_hash}"
        )

    for name, split in dataset.items():
        expected = manifest["counts_by_split"].get(name)
        if expected != len(split):
            failures.append(f"split '{name}': expected {expected} rows, found {len(split)}")

    seen: set[str] = set()
    checked = 0
    for name, split in dataset.items():
        for record in split:
            example_id = record["example_id"]
            if example_id in seen:
                failures.append(f"duplicate example_id: {example_id}")
            seen.add(example_id)

            prompt = record["prompt"]
            if not (
                isinstance(prompt, list)
                and len(prompt) == 2
                and prompt[0]["role"] == "system"
                and prompt[1]["role"] == "user"
            ):
                failures.append(f"{example_id}: malformed prompt")
                continue
            if prompt[1]["content"] != record["question"]:
                failures.append(f"{example_id}: prompt/question mismatch")

            if record["task_family"] in ("count", "index"):
                if not record["target_json"]:
                    failures.append(f"{example_id}: missing target_json")
                    continue
                target = json.loads(record["target_json"])
                leak = _leaks_target(record["question"], target)
                if leak:
                    failures.append(f"{example_id}: leakage ({leak})")
                # Recompute from the molecule in a fresh process: this is the
                # check that a stored answer still describes its question.
                recomputed = recompute_target(
                    record["question_smiles"],
                    json.loads(record["properties_json"]),
                    record["task_family"],
                    engine,
                )
                if recomputed != target:
                    failures.append(
                        f"{example_id}: stored target {target} != recomputed {recomputed}"
                    )
                score = _reoracle(record["task_type"], target=target)
            else:
                if not record["constraints_json"]:
                    failures.append(f"{example_id}: missing constraints_json")
                    continue
                score = _reoracle(
                    "constraint_generation",
                    constraints=json.loads(record["constraints_json"]),
                    witness=record["witness_smiles"],
                )
            if score != 1.0:
                failures.append(f"{example_id}: oracle re-check failed ({score})")
            checked += 1

    print(f"verified {checked} examples across {len(dataset)} splits")
    if failures:
        print(f"\nFAILED with {len(failures)} problem(s):")
        for failure in failures[:40]:
            print(f"  - {failure}")
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")
        sys.exit(1)
    print("all invariants hold; artifact is ready for training")


def _reoracle(
    task_type: str,
    target: dict[str, Any] | None = None,
    constraints: list[dict[str, Any]] | None = None,
    witness: str | None = None,
) -> float:
    from moleculariq_core import evaluate_answer

    if constraints is not None:
        return float(
            evaluate_answer(
                task_type="constraint_generation",
                predicted=json.dumps({"smiles": witness}),
                constraints=constraints,
            )
        )
    return float(
        evaluate_answer(
            task_type=task_type, predicted=json.dumps(target), target=target
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="generate and freeze the dataset")
    build_parser.add_argument("--config", type=Path, required=True)
    build_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing artifact (off by default: artifacts are immutable)",
    )

    verify_parser = sub.add_parser("verify", help="reload and re-check an artifact")
    verify_parser.add_argument("--artifact", required=True)

    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.config, overwrite=args.overwrite)
    else:
        verify(args.artifact)


if __name__ == "__main__":
    main()
