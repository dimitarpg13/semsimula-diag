# semsimula-diag

Diagnostic instrumentation for Fock-PARFLM / SemSimula training runs —
gradient-spike capture, deterministic replay forensics, and curvature/stiffness
probes, extracted from the CfC/BAOAB training notebooks so they can be
imported, versioned and tested instead of living as notebook cells.

Design follows `companion_notes/Diagnostic_Programme_in_CfC_BAOAB_Integrator.md`
§11 in the [semsimula-paper](https://github.com/dimitarpg13/semsimula-paper)
repo. See [MIGRATION.md](MIGRATION.md) for what is ported so far.

> ## Verification status — library port vs notebook original
>
> **The methods here are not untested — the *library port* of some of them is.**
> These probes were extracted from the CfC/BAOAB training notebooks, where they
> did the work they were built for: the inline `Cell 6d` gradient-spike replay
> forensics produced the energy/probability spike diagnosis that motivated the
> move away from Velocity-Verlet, and stiffness and resonance monitoring ran
> throughout the CfC+BAOAB runs. What the table tracks is whether the
> *extracted module* has been shown to reproduce that behaviour, on GPU,
> against known-good values.
>
> | module | port verified? |
> |---|---|
> | `clipping`, `replay`, `report` | **yes** — bit-equality parity vs the notebook original |
> | `probes.stiffness` | **yes** — CPU against real step-87196 weights; `sigma_lr_spectrum_by_site` GPU-verified at step 96410 |
> | `probes.resonance` | **yes** — GPU, steps 87196 / 86201 / 90360; max `omega*dt` 1.50–1.52, 0.000% of tokens over the wall; power iteration cross-checked against `eigvalsh` on CPU. **But see the open caveat below.** |
> | `probes.precision_cap.replay_rank_truncation_ablation` | **yes** — GPU, step 87196; the `rank=4` arm reproduces the untruncated reference exactly (2539.20, error 0.0000) |
> | `probes.tau_saturation` | **yes** — CPU; `tau_min` 5.22 @ register 14 matches the live training log |
> | `probes.tokens` | **yes** — CPU, against real bundles |
> | `probes.generator_rank` | exercised from 4 CfC+BAOAB notebooks; no known-good reference recorded |
> | `probes.layer_profile` | **port unverified** — needs a GPU parity run |
> | `probes.row_attribution` | **port unverified** — needs a GPU parity run |
> | `probes.precision_cap` (other entry points) | **port unverified** — needs a GPU parity run |
> | `probes.clip_order` | **port unverified** — needs a GPU parity run |
> | `probes.integrator` | **port unverified** — needs a GPU parity run |
> | `probes.tau_saturation.probe_hot_rows` | **port unverified** — ported after the rest; same gap |
> | `probes.precision_cap.replay_rank_perturbation_control` | **port unverified** — built as the control for the truncation ablation |
>
> **Why GPU matters for the replay-based rows.** A replay is only meaningful if
> it reproduces the gradients training actually saw. That was confirmed for the
> notebook's own `replay_spike_batch` on an A100 (fidelity 0.0002% at step
> 87196, 0.0003% at 90360) — but a CPU replay of the same bundle does **not**
> reproduce it, for reasons not yet root-caused. A green CPU test therefore says
> nothing about replay correctness, and an unverified row above means *"the
> notebook version of this is trusted; this module has not been shown to match
> it on GPU"* — not that the diagnostic itself is in doubt.
>
> **Open caveat on `probes.resonance` — 2026-09-26.** Although GPU-verified at
> the steps above, the wall monitor printed
> `EMPTY SUMMARY -- no substep hook fired` throughout the 2026-09 depth-ladder
> runs (L=1, L=2 `'none'`, and the conservative-only arm). That is a **missing**
> diagnostic, not a passing one — the substep hook is not matching the path
> `baoab_cfc_lowrank` actually calls. Until it is root-caused, resonance output
> from those runs should be read as absent rather than clean.
>
> **To verify a port row:** run the verification section at the end of
> [`examples/notebooks/semsimula_diag_tour.ipynb`](examples/notebooks/semsimula_diag_tour.ipynb)
> on a GPU runtime. It checks each replay probe against the known-good values
> above and prints a pass/fail table.

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
