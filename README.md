# Single-task GRPO on MolecularIQ

Train `Qwen/Qwen2.5-0.5B-Instruct` with TRL's GRPO on **one MolecularIQ task
family at a time**, then compare all three specialists against the untouched
base model on the whole official benchmark.

| Model | Trained on | Checkpoint |
|---|---|---|
| baseline | nothing | `Qwen/Qwen2.5-0.5B-Instruct` |
| count | counting questions only | `runs/grpo-count-r001/final` |
| index | index-attribution questions only | `runs/grpo-index-r001/final` |
| constraint | constrained-generation questions only | `runs/grpo-constraint-r001/final` |

All four are scored by the official `moleculariq-eval` harness on all 5,111 test
items. Nothing else counts as a result.

---

## The shape of the thing

```
MolecularIQ training pool (1.27M molecules, HF: ml-jku/moleculariq-trainPool)
        │
        │  CPU, once, offline           src/miqgrpo/generation.py
        ▼                               src/miqgrpo/build_dataset.py
frozen dataset artifact  data/processed/miq-train-v001/
   dataset/ + manifest.json + preprocessing_config.yaml + provenance.json
   splits: count · index · constraint_generation (+ *_dev)
        │
        │  GPU, three separate runs     src/miqgrpo/train_grpo.py
        ▼                               src/miqgrpo/rewards.py
runs/<experiment_id>/
   frozen_config.yaml · provenance.json · logs/metrics.jsonl · checkpoints/ · final/
        │
        │  GPU, four separate runs      src/miqgrpo/evaluate.py
        ▼
results/moleculariq/<evaluation_run_id>/
   eval_manifest.json · summary.json · raw/ (official harness output)
        │
        ▼                               src/miqgrpo/plots.py
results/figures/
```

Each arrow is a separate command writing an immutable artifact. A failure in one
stage never forces redoing the others.

---

## Quick start

```bash
bash scripts/00_setup_env.sh          # env + official packages     (network)
python scripts/01_prefetch_assets.py  # model + datasets into cache (network)
bash scripts/02_build_dataset.sh      # frozen dataset              (CPU, ~15 min)
bash scripts/10_train_all.sh          # three GRPO runs             (GPU)
bash scripts/20_evaluate_all.sh       # four benchmark runs         (GPU)
bash scripts/30_make_plots.sh         # report figures
```

`scripts/run_all.sh` chains the last four. For the RunPod A100 walkthrough
(volumes, offline mode, wall-clock, what to check at each step) see
[docs/runbook-runpod.md](docs/runbook-runpod.md).

Before any long GPU run:

```bash
python -m miqgrpo.train_grpo preflight --config configs/experiments/grpo-count-r001.yaml
```

Preflight resolves the batch arithmetic, verifies the dataset hash, prints the
fully rendered chat prompt, generates a real rollout and shows each completion
with its parsed answer and reward components. It is the cheapest place to catch
the failures that otherwise surface an hour into a paid allocation.

---

## Design decisions worth knowing

### Questions are generated before training, never inside it

`build_dataset.py` writes every question, target and constraint to disk, hashes
the result, and `train_grpo.py` loads that and nothing else — it does not import
the generator. Generating questions inside the optimisation loop would make the
data a moving target and the run unreproducible.

Every generated row is checked by the **official verifier against its own stored
answer** before it enters the dataset, and count/index targets are additionally
**recomputed from the molecule** during verification. A row whose answer the
verifier rejects is a row GRPO could never earn reward on; it would just add
noise to the gradient.

### Targets follow the SMILES the question displays

Half the questions show a randomised or kekulised rendering of the molecule.
Atom indices depend on that string's ordering, so targets are computed on the
displayed string, not on the canonical form. This matches the official
convention — verified directly against official test items, see
[docs/official-semantics.md](docs/official-semantics.md).

### Questions have to be non-trivial

Three quarters of the property space is functional groups, and most molecules
have none of any particular one. Sampling naively produces a dataset where the
answer is almost always `0` or `[]` and "generate a molecule with no boc group"
is satisfied by literally anything. A policy learns the constant, not the
chemistry. So:

* categories get equal budget (not properties), as in the official generator;
* molecules are drawn with inverse-frequency weighting over property values,
  with zero-valued molecules down-weighted;
* a generation constraint set satisfied by more than **3%** of reference
  molecules is rejected outright.

### The reward is the official one, plus a small shaping term

```
total = 1.0 × official_verifier(extract(completion), stored_target)
      + 0.1 × answer_shape
      + 0.05 × valid_smiles          (constrained generation only)
```

Extraction uses `moleculariq-eval`'s own `extract_moleculariq_answer`, vendored
byte-for-byte into `src/miqgrpo/vendor/`, and scoring uses
`moleculariq_core.evaluate_answer`. Training reward and benchmark score are
therefore the same measurement.

The shaping term exists because of a measured problem: the base policy answers
essentially no chemistry question correctly *and* writes a ```json fence instead
of the `<answer>` block the prompt asks for. Both reward terms were flat zero
across every rollout in a group, so GRPO had no advantage to propagate at all.
Grading the shape term into four pieces — envelope, valid JSON object, the exact
keys the question named, plausible value types — took zero-variance groups from
5/6 to 0/6 in the preflight rollout. It never reveals an answer (the keys are
printed in the question) and is capped far below correctness, so a well-shaped
wrong answer can never outrank a correct one. `tests/test_rewards.py` asserts
exactly that.

### There is a dev split, and it is not the benchmark

Each family also gets a `*_dev` split — 5% of the generated examples, held out
from training. It exists so generalisation *within the training distribution*
can be watched without touching the official benchmark. It is off by default
(`runtime.eval_steps: null`) because a GRPO eval pass costs a full round of
generation; set `eval_steps` to switch it on.

It is a debugging aid, not a result. No checkpoint is selected with it either —
the reported checkpoint is simply the final one.

### The benchmark is test-only

- Preprocessing refuses to start if a benchmark or hidden-pool identifier appears
  as an input.
- `evaluate.py` refuses `--limit` unless `--smoke` is passed, and a smoke run is
  written to a separate directory stamped `full_benchmark: false`.
- Every experiment config and eval manifest carries explicit
  `benchmark_used_for_*: false` fields. They are claims; keep them true.
- Nothing in this repository selects a checkpoint, a hyperparameter or a prompt
  using a benchmark score. The one benchmark-informed decision made during
  development — confirming the atom-indexing convention — is a task *definition*,
  not an outcome, and is documented in `docs/official-semantics.md`.

---

## Layout

```
configs/preprocessing/miq-train-v001.yaml   dataset recipe (immutable)
configs/experiments/grpo-*-r001.yaml        one per trained model (immutable)

src/miqgrpo/
  generation.py      offline question generation + property probing
  build_dataset.py   freeze, validate, manifest, verify
  prompts.py         the system+user turns, identical to the benchmark's
  rewards.py         runtime scoring + diagnostics (TRL reward callables)
  train_grpo.py      preflight and training
  evaluate.py        official benchmark runner
  plots.py           report figures
  config.py          experiment config schema
  paths.py           env-driven roots ($MIQ_DATA, $MIQ_RUNS, $MIQ_RESULTS)
  provenance.py      git state, versions, hashes
  vendor/            verbatim copies of official eval code + VENDOR.json

scripts/             the pipeline, numbered in running order
scripts/slurm/       Leonardo sbatch equivalents
tests/               reward-hacking, prompt-drift and dataset-integrity tests
```

Redirect the heavy directories with environment variables:

```bash
export MIQ_DATA=/workspace/miq/data
export MIQ_RUNS=/workspace/miq/runs
export MIQ_RESULTS=/workspace/miq/results
export HF_HOME=/workspace/miq/hf
```

---

## Tests

```bash
pytest -q                          # 106 tests, ~4 s, no GPU
python scripts/check_vendor.py     # vendored official code still matches upstream
```

`tests/test_rewards.py` covers the reward-hacking list: wrong counts, off-by-one
indices, empty and truncated output, conflicting answer blocks, invalid SMILES,
valid molecules that violate a constraint, different valid molecules that satisfy
one, pretty formatting with wrong chemistry, and a verifier that raises.

### What has already been run, and what has not

Validated on CPU before this was handed over:

- the full-scale dataset build (24,000 examples from the real 1.27M-molecule
  pool) and its fresh-process verification;
- a real GRPO training loop through `GRPOTrainer` — optimizer steps, reward
  diagnostics reaching `metrics.jsonl`, checkpoint written and reloadable;
- stop and resume from a mid-run checkpoint, continuing at the right step;
- the preflight rollout, end to end, against the base model;
- all figures, against synthetic metrics (then deleted).

Not yet run, because it needs the GPU and the official harness installed: the
three full training runs, and the four benchmark evaluations. `--dry-run` shows
the exact `lm_eval` command the evaluation stage will execute.

---

## Figures

| File | What it shows |
|---|---|
| `training_curves.png` | reward, verifier correctness, answer-tag compliance, completion length per run |
| `benchmark_headline.png` | pass@1 / pass@3 / avg accuracy, four models |
| `benchmark_by_task_type.png` | the transfer question: does count-training help index? |
| `transfer_matrix.png` | change vs base model, trained-on × evaluated-on |
| `benchmark_by_complexity.png` | by Bertz complexity bin and multitask load |
| `benchmark_summary.csv` | the same numbers as a table |

Training reward and benchmark accuracy are never drawn on the same axes — they
are different quantities measured on different data.

For qualitative examples in the report, TRL writes every logged rollout to
`runs/<id>/checkpoints/completions/*.parquet` — prompt, completion, each reward
component, and the `parse_status` / `format_status` / `extracted_answer` columns
this project adds:

```python
import pandas as pd
df = pd.read_parquet("runs/grpo-count-r001/checkpoints/completions/completions_00001.parquet")
print(df[["prompt", "completion", "parse_status", "extracted_answer"]].head())
```

---

## Upstream

| Repository | Commit | Used for |
|---|---|---|
| [moleculariq-core](https://github.com/ml-jku/moleculariq-core) | `a1b8963` | question generation, symbolic solver, official verifier |
| [moleculariq-eval](https://github.com/ml-jku/moleculariq-eval) | `425ecaa` | official benchmark harness, answer extraction, system prompt |
| [moleculariq-benchmark](https://github.com/ml-jku/moleculariq-benchmark) | `cf08baa` | reference for how official questions are constructed |

Paper: [MolecularIQ (ICLR 2026)](https://arxiv.org/abs/2601.15279).
