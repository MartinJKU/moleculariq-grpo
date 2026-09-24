# Runbook: RunPod A100 PCIe

End-to-end on a single A100. Written for a pod with one GPU and a persistent
volume mounted at `/workspace`.

Budget roughly: dataset ~15 min CPU, training 4–6 h per model at the shipped
settings, benchmark 1–2 h per model with vLLM. Four benchmark runs, three
trainings.

---

## 0. Pod

- Template: any recent PyTorch CUDA image (torch ≥ 2.6).
- GPU: 1 × A100 PCIe (40 GB is enough; 80 GB gives headroom for vLLM).
- Volume: ≥ 100 GB at `/workspace`.

Everything heavy goes on the volume — the container filesystem does not survive
a pod restart.

```bash
export MIQ_ROOT=/workspace/miq
export MIQ_DATA=$MIQ_ROOT/data
export MIQ_RUNS=$MIQ_ROOT/runs
export MIQ_RESULTS=$MIQ_ROOT/results
export HF_HOME=$MIQ_ROOT/hf
mkdir -p "$MIQ_DATA" "$MIQ_RUNS" "$MIQ_RESULTS" "$HF_HOME"
```

Put those five lines in `~/.bashrc` now. Forgetting them writes 30 GB of
checkpoints into the container layer, and you find out when the pod restarts.

---

## 1. Install (needs network)

```bash
cd /workspace
git clone <your fork> moleculariq-grpo && cd moleculariq-grpo
bash scripts/00_setup_env.sh
source .venv/bin/activate
```

This clones `moleculariq-core` and `moleculariq-eval` at pinned commits into
`third_party/` and installs both editable. `moleculariq-eval` *is* `lm_eval`:
do not also install upstream `lm-evaluation-harness`, or whichever wins on
`sys.path` decides how the benchmark is scored.

Check:

```bash
pytest -q
python scripts/check_vendor.py --fetch
```

---

## 2. Stage assets (needs network — do it before the GPU phases)

```bash
python scripts/01_prefetch_assets.py
```

Downloads the policy, the training pool and the official benchmark into
`$HF_HOME`, then re-resolves all three with `HF_HUB_OFFLINE=1` and fails if any
of them cannot be loaded offline. Everything after this point runs with the Hub
switched off, so a miss here is a crash minutes into a GPU allocation.

Writes `assets_manifest.json` with the resolved revision of each.

---

## 3. Dataset (CPU, ~15 min)

```bash
bash scripts/02_build_dataset.sh
```

Three steps: build, verify in a **fresh process**, then pin the artifact's
content hash into all three experiment configs.

Worth reading afterwards:

```bash
python - <<'PY'
import json
m = json.load(open("data/processed/miq-train-v001/manifest.json"))
print("examples      ", m["num_examples"])
print("molecules     ", m["num_unique_molecules"])
print("hash          ", m["dataset_hash"])
print("pool revision ", m["source_dataset_revision"])
print("drops         ", json.dumps(m["drop_counts_by_reason"], indent=2))
PY
```

Expect a substantial `constraint_too_common` drop count — that is the prevalence
filter refusing constraints that any molecule satisfies, and it is working.

---

## 4. Training (GPU, three runs)

Preflight first. It is cheap and it is where problems show up:

```bash
python -m miqgrpo.train_grpo preflight --config configs/experiments/grpo-count-r001.yaml
```

Read four things in the output:

| Line | What bad looks like |
|---|---|
| batch arithmetic | any `INVALID:` line — fix before the model loads |
| rendered prompt | anything other than system=official prompt, user=bare question |
| per-group `std` | `0.000` on every group means GRPO gets no gradient |
| completion length max | at the cap means answers are being truncated |

Then:

```bash
bash scripts/10_train_all.sh
# or one at a time:
python -m miqgrpo.train_grpo train --config configs/experiments/grpo-count-r001.yaml
```

Watch `runs/<id>/logs/metrics.jsonl`:

```bash
tail -f runs/grpo-count-r001/logs/metrics.jsonl | python -c "
import json,sys
for line in sys.stdin:
    d = json.loads(line)
    print(f\"{d.get('step'):>5} reward={d.get('reward',0):.3f} \"
          f\"correct={d.get('reward/correctness_mean',0):.3f} \"
          f\"tags={d.get('parse/answer_tag_fraction',0):.2f}\")
"
```

Healthy early behaviour: `tags` climbs first (the shape term is the only signal
the policy can act on at the start), `correct` follows more slowly. If `reward`
rises while `correct` stays flat for a long stretch, the policy is collecting
shape credit and not learning chemistry — inspect logged completions before
letting it run further.

Resume after a pod restart:

```bash
python -m miqgrpo.train_grpo train --config configs/experiments/grpo-count-r001.yaml --resume
```

### If rollouts are too slow

The shipped config uses plain `model.generate`. vLLM colocate is roughly 3× faster
for 0.5B rollouts:

```yaml
runtime:
  use_vllm: true
  vllm_mode: colocate
  vllm_gpu_memory_utilization: 0.3
```

Treat it as a behaviour change: new experiment ID, and re-run preflight, because
the sampling path is no longer the same one that produced the earlier numbers.

---

## 5. Benchmark (GPU, four runs)

```bash
bash scripts/20_evaluate_all.sh
# vLLM unavailable / version clash:
BACKEND=hf bash scripts/20_evaluate_all.sh
```

Infrastructure check first if you want one — it is clearly marked and can never
be quoted as a result:

```bash
python -m miqgrpo.evaluate run --run-id smoke --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --label baseline --smoke
```

Each real run writes `results/moleculariq/<id>/` with the manifest, the harness's
own results file, per-sample logs, and `output.log` (the harness's own output,
which is also streamed to your terminal as it runs). Result directories are
never overwritten; a repeat needs a new `--run-id`.

---

## 6. Figures

```bash
bash scripts/30_make_plots.sh
ls results/figures/
```

---

## Troubleshooting

**`generation_batch_size is not divisible by num_generations`** — preflight says
this before anything loads. `per_device_train_batch_size × gradient_accumulation_steps
× world_size` must divide by `num_generations`.

**`dataset artifact ... does not match the hash`** — the dataset was rebuilt after
the config was pinned. Either restore the artifact or build a new one with a new
`dataset_artifact_id` and a new experiment ID. Do not just re-pin: that would
silently change what the experiment ID means.

**`run directory ... already exists and is not empty`** — deliberate. Use a new
experiment ID, or `--resume` if you are continuing that run.

**CUDA OOM during training** — lower `per_device_train_batch_size` and raise
`gradient_accumulation_steps` by the same factor to keep the generation batch
unchanged. `auto_find_batch_size` is rejected by TRL for GRPO because shrinking
the batch would break prompt grouping.

**Every group has zero reward variance** — the policy answers identically across
rollouts. Check `temperature`, check the logged completions, and check that the
shape reward is actually enabled (`format_weight > 0`).

**`ModuleNotFoundError: No module named 'ray'` at evaluation** — the harness's
vLLM backend imports ray at module level, but vLLM only depends on it for
multi-GPU. `pip install ray`, delete the empty result directory, re-run.
`scripts/00_setup_env.sh` installs it alongside vLLM now.

**vLLM refuses to install / import** — skip it. `INSTALL_VLLM=0 bash
scripts/00_setup_env.sh` and `BACKEND=hf` for evaluation. Slower, same numbers.
