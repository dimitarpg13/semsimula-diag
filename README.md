# semsimula-diag

Diagnostic instrumentation for Fock-PARFLM / SemSimula training runs —
gradient-spike capture, deterministic replay forensics, and curvature/stiffness
probes, extracted from the CfC/BAOAB training notebooks so they can be
imported, versioned and tested instead of living as notebook cells.

Design follows `companion_notes/Diagnostic_Programme_in_CfC_BAOAB_Integrator.md`
§11 in the [semsimula-paper](https://github.com/dimitarpg13/semsimula-paper)
repo. See [MIGRATION.md](MIGRATION.md) for what is ported so far.

> ## ⚠️ GPU verification pending
>
> All notebook probe families are now ported, but **the replay-based probes
> have not yet been verified against the live model on an A100**:
>
> | module | status |
> |---|---|
> | `probes.layer_profile` | **unverified** — needs GPU |
> | `probes.row_attribution` | **unverified** — needs GPU |
> | `probes.precision_cap` | **unverified** — needs GPU |
> | `probes.clip_order` | **unverified** — needs GPU |
> | `probes.integrator` | **unverified** — needs GPU |
> | `probes.stiffness` | verified on CPU against the real step-87196 weights |
> | `probes.tau_saturation` | verified on CPU (`tau_min` 5.22 @ register 14 matches the live log) |
> | `probes.tokens` | verified on CPU against real bundles |
> | `clipping`, `replay`, `report` | verified; bit-equality parity tests vs the notebook original |
>
> **Why GPU matters here.** A replay is only meaningful if it reproduces the
> gradients training actually saw. That was confirmed for the notebook's own
> `replay_spike_batch` on an A100 (fidelity 0.0002% at step 87196, 0.0003% at
> 90360) — but a CPU replay of the same bundle does **not** reproduce it, for
> reasons not yet root-caused. So a green CPU test says nothing about
> replay correctness.
>
> **To verify:** run the verification section at the end of
> [`examples/notebooks/semsimula_diag_tour.ipynb`](examples/notebooks/semsimula_diag_tour.ipynb)
> on a GPU runtime. It checks each replay probe against the known-good values
> above and prints a pass/fail table. Until that passes, treat every
> replay-based probe's output as unconfirmed.

## Install

```bash
pip install -e ".[test]"
pytest -q
```

## The four diagnostic phases

The programme organises instruments as a pipeline of increasing cost and
specificity — the cheap end mines what the training loop already logs, the
expensive end reconstructs one offending step bit-for-bit.

| phase | cost | question it answers | module |
|---|---|---|---|
| 0 | always on, ~free | *when and how often?* | `phase0` |
| 1 | on trigger, cheap | *which exact step?* | `capture` |
| 2 | offline, exact | *where inside the model?* | `replay` + `probes/` |
| 3 | productionized | reusable, tested probe | `report` |

## Usage

Per-group clipping, and both clip strategies:

```python
from semsimula_diag import GradClipConfig, ClipThenSum, clip_grads_per_group

cfg = GradClipConfig(
    default_clip=1.0,
    # ORDER MATTERS: a more specific key must precede a less specific one
    # that also matches ('creation_gate_qkv.log_tau' matches both below).
    overrides={'log_tau': 0.3, 'creation_gate': 0.3, 'depth_code': 0.25},
    watchdog_exclude_groups=frozenset({'override:reverse_channel_scale'}),
    clip_then_sum_groups=frozenset({'E', 'P'}),
    clip_then_sum_threshold=0.5,
)

cts = ClipThenSum(model, cfg)
for xb, yb in microbatches:
    (loss_fn(model(xb), yb) / grad_accum).backward()
    cts.apply_microbatch()
cts.splice_back()

total_norm, per_group = clip_grads_per_group(model, cfg)
```

Loading a capture bundle, with automatic fallback to the permanent archive
when the live ring has rotated it out:

```python
from semsimula_diag import BundleStore, isolated_grads

store = BundleStore(ckpt_dir=CKPT_DIR, ckpt_prefix=CKPT_PREFIX,
                    archive_root=GDRIVE_ROOT)
bundle, path = store.load(87196)

with isolated_grads(model):        # probe without disturbing live gradients
    ...
```
