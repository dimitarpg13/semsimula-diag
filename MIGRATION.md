# Migration status

Tracks the port of the CfC/BAOAB notebook diagnostics into `semsimula_diag`,
following the module layout in `semsimula-paper`'s
`companion_notes/Diagnostic_Programme_in_CfC_BAOAB_Integrator.md` §11.2–§11.3
and the low-risk-first ordering in §11.6.

**Nothing is removed from `semsimula-paper`.** This repo duplicates the
functionality with imports/paths adjusted; the notebook cells and
`scaleup/*.py` utilities stay exactly as they are until a later pass switches
them over to importing from here.

## §11.6 order

| # | step | status |
|---|---|---|
| 1 | `grad_clip_utils.py` → `clipping` | **done** |
| 7 | `clip_then_sum` splice → `clipping` | **done** (pulled forward — same module, and the two clip strategies are only meaningful side by side) |
| 2 | replay primitives → `replay` | **partial** — `grad_snapshot`/`grad_restore`/`isolated_grads` and `BundleStore` done; the three probes themselves not yet |
| — | `ProbeResult` (§11.4) | **done** — every probe depends on it, so it lands before them |
| 3 | ablation helpers + stiffness family | **done** — `precision_cap`, `clip_order`, `integrator`, `stiffness` |
| 4 | `tau_saturation` / `tokens` family | **done** — both |
| 5 | Phase-0 writers → `phase0` | not started |
| 6 | capture watchdog + checkpoint I/O → `capture` | not started |
| 8 | fold SCAF `GradientSpikeProbe` onto `probes/` | not started |

**All notebook probe families are ported.** What remains is `phase0`
(the JSONL writers), `capture` (the watchdog and checkpoint I/O), and the
SCAF adoption — none of which are probes.

## What is here

```
src/semsimula_diag/
├── clipping.py   GradClipConfig, assign_clip_group, per_group_grad_norms,
│                 clip_grads_per_group, clip_then_sum_params, ClipThenSum
├── replay.py     grad_snapshot, grad_restore, isolated_grads, BundleStore
├── report.py     ProbeResult
└── probes/
    ├── context.py         ProbeContext, MissingContextError
    ├── _engine.py         replayed(), restored_model_state(), ReplayInfo
    ├── layer_profile.py   replay_spike_batch
    ├── row_attribution.py attribute_spike_rows
    ├── precision_cap.py   replay_precision_cap_ablation,
    │                      replay_curvature_rebalance_ablation,
    │                      replay_rank_truncation_ablation
    ├── clip_order.py      replay_clip_ablation
    ├── integrator.py      replay_integrator_ablation
    ├── tau_saturation.py  probe_gate_saturation, sweep_log_tau_history,
    │                      probe_hot_rows
    ├── stiffness.py       stiffness_report, sigma_lr_report,
    │                      sigma_lr_spectrum_report, bracket_precision_lr_max,
    │                      spectrum_across_checkpoints
    └── tokens.py          row_degeneracy, inspect_spike_tokens,
                           decode_hot_rows
```

The six replay-based probes share `_engine.replayed()`, which writes the
snapshot / load-weights / restore-RNG / microbatch-loop / clip-then-sum /
restore sequence **once**. In the notebook each helper re-implements it, and
getting any step subtly wrong changes the answer without raising.

`probe_hot_rows` needed two more primitives, now also in `_engine.py`:
`patched_attrs` (restoring-guaranteed multi-attribute monkeypatching, for
hooking the creation gate's module-level readout functions) and
`iter_isolated_rows` (the per-row RNG-reset-and-clear loop, shared with
`row_attribution.attribute_spike_rows`'s own per-row pass). Both are general
enough that a later probe needing the same pattern does not have to
reinvent it.

### ⚠️ Verification status

| module | verified? |
|---|---|
| `clipping`, `replay`, `report` | yes — bit-equality parity vs the notebook original |
| `probes.tokens` | yes — CPU, against real bundles |
| `probes.stiffness` | yes — CPU, against real step-87196 weights |
| `probes.tau_saturation` | yes — CPU; `tau_min` 5.22 @ register 14 matches the live training log |
| `probes.layer_profile` | **NO — needs GPU** |
| `probes.row_attribution` | **NO — needs GPU** |
| `probes.precision_cap` | **NO — needs GPU** |
| `probes.clip_order` | **NO — needs GPU** |
| `probes.integrator` | **NO — needs GPU** |
| `probes.tau_saturation.probe_hot_rows` | **NO — needs GPU** (ported later than the rest of this table; same engine, same gap) |
| `probes.precision_cap.replay_rank_truncation_ablation` | **NO — needs GPU** (built 2026-09-13, was proposed-only in the companion note until now; same gap). First GPU attempt aborted after 4 h at 3/5 arms — see the note below |

### Rank truncation must not pay the `baoab_cfc_lowrank` tax

The first GPU run of `replay_rank_truncation_ablation` (step 87196, A100,
2026-09-13) was killed after 4 hours having completed only 3 of its 5
arms. The cause was in `_svd_truncate`, which called
`torch.linalg.svd(B)` directly: `B` is `(K, d, r)` **per token**, so at
`B=8, T=512, K=16, n_ctx=5, L=8` one forward pass issues on the order of
2.6M batched `384 x 4` decompositions, and the backward differentiates
through every one of them.

That is the *same* operation, at the same scale, that makes the
`baoab_cfc_lowrank` integrator arm non-viable — `parf/cfc_baoab.py`
measures that arm at ~120 s/step vs `baoab_cfc`'s ~10-15 s/step and
states plainly that "the batched per-token SVD is the entire extra cost".
A probe built to *diagnose* the low-rank channel had silently taken on
the cost profile of the arm it diagnoses.

Two corrections, both in `_svd_truncate`:

1. **Don't route it to CPU.** An initial fix moved the SVD to `B.cpu()`.
   That is backwards: `cfc_baoab.py` engineers a jitter + retry ladder
   (`_LOWRANK_SVD_JITTER`, `_LOWRANK_SVD_MAX_TRIES`) specifically to keep
   one ill-conditioned batch element from dragging the batch onto the CPU
   LAPACK fallback, "which, called per layer per step, is the dominant
   wall-clock cost". Reverted.
2. **Use the Gram matrix and a detached projector.** Since `d >> r`,
   truncation needs only the right singular subspace: `B = U S V^T` gives
   `B V_{r'} V_{r'}^T` as the exact rank-`r'` truncation, and `V` comes
   from the `r x r` Gram matrix `B^T B` — a 4x4 `eigh` instead of a
   384x4 SVD, with `U` never formed. The projector is computed under
   `no_grad`, so the truncation is the linear map `B @ P` and *nothing*
   differentiates through the decomposition.

Verified equal to the literal SVD truncation to 1e-13 (double), with the
achieved rank exactly `r'`. Point 2 also removes a correctness risk, not
just a cost one: the backward's `1/(sigma_i^2 - sigma_j^2)` terms are
gone, and zeroing the tail *manufactures* the repeated singular values
that `cfc_baoab.py` records as making cuSOLVER "return NaN singular
vectors/values *without raising*". The aborted run's `rank=1` and
`rank=2` arms both reported `relative_force_error` pinned at ~1.0000 with
the gradient norm collapsed 2539 -> ~2.9; those numbers were discarded
rather than trusted, and the ablation should be re-run from scratch.

### Interrupting a replay poisons the kernel (`assert_unpatched`)

The follow-up attempt surfaced a second, independent hazard. After the
4-hour run was interrupted, a `git pull` + re-run produced a
`CheckpointError` on the *untruncated reference* arm — the one arm that
uses `nullcontext()` and never calls `_svd_truncate` at all. The saved
metadata was `[8,512,8,4]`, `[8,512,8,384,4]`, `[8,512,8,4,4]` (exactly
`S`, `U`, `Vh` for `B` of shape `(8,512,K=8,d=384,r=4)`) against
recomputed `[8,512,8,384]` throughout: the forward had run an SVD and the
backward recompute had not.

Two causes, both worth knowing:

1. **`git pull` does not reload an imported module.** The kernel kept
   serving the cached `sys.modules` entry, so the re-run still executed
   the old GPU-SVD code. Only a kernel restart (or an explicit
   `importlib.reload`) picks up pulled source.
2. **`KeyboardInterrupt` can orphan `patched_attrs`.** It is a
   generator-based context manager; interrupting mid-`yield` leaves the
   generator suspended rather than closed, so its `finally` restore runs
   whenever CPython later garbage-collects it. Here that landed between
   a gradient-checkpointed forward and its backward recompute, which is
   why the two disagreed.

The `CheckpointError` was the lucky outcome. Had the collection landed a
moment later, the run would have *succeeded* with a silently rank-3
truncated "untruncated" reference — rescaling every `relative_force_error`
measured against it, with nothing to indicate anything was wrong.

`patched_attrs` now records active patch names on the patched object, and
`assert_unpatched(obj, *names)` turns a leaked patch into an immediate
`RuntimeError` naming the restart requirement.
`replay_rank_truncation_ablation` calls it before doing anything else.
The record lives on the object, not on the wrapper, because a
`make_wrapper` may legitimately return a bound method or other callable
that rejects attribute assignment.

The replay probes have unit tests covering the engine (weight/grad/RNG
restoration, restoration on the exception path, clip-then-sum actually
firing) against a toy model — but **a passing CPU test says nothing about
whether a replay reproduces the real gradients**, since a CPU replay of a
real bundle demonstrably does not, for reasons not yet root-caused.

Run Part II of `examples/notebooks/semsimula_diag_tour.ipynb` on a GPU
runtime to close this out. It checks each replay probe against the
known-good A100 values (fidelity 0.0002% at step 87196, 0.0003% at 90360,
plus exact per-group norms) and prints a pass/fail table.

## Testing

`tests/test_parity_with_notebook.py` is the §11.5 gate: it imports the
original `grad_clip_utils` from a sibling `semsimula-paper` checkout and
asserts bit-equality on identical inputs, including the `log_tau` /
`creation_gate` override-ordering trap from Mitigations §49.8. It skips
cleanly when that checkout is absent (`SEMSIMULA_PAPER` overrides the path).

### On the §11.5 golden-output fixtures

The archived probe outputs at steps 70,522 / 71,194 / 71,703 **cannot** be
turned into regression tests: their `*_spikebatch.pt` bundles have been
evicted from the rotating ring and are gone (the oldest surviving archived
bundle is step 77,223). The printed outputs survive, but without their
inputs they stay documentation-level golden values.

That costs less than it first appears. Golden values only prove "matches a
number recorded some time ago"; the stronger property — and the one §11.5
actually asks for — is that the ported code and the notebook original agree
on *the same input*. That is what `test_parity_with_notebook.py` asserts,
and it needs no particular historical step.

Real bundles now in use (located at run time via `SEMSIMULA_BUNDLES`, never
committed — they are ~294 MB each):

| step | role |
|---|---|
| 87196 | register-decoupled tail event (reg/V_theta = 8.59); the §1 priority checkpoint |
| 90360 | V_theta-slaved contrast case (reg/V_theta = 0.63) |

`tests/test_real_bundles.py` pins their recorded pre-clip norms and ratios
and checks the bundle shape every probe depends on. Step 85,885 exists only
as a `_prereload` (weights, no batch or RNG), so it is usable by
`spectrum_across_checkpoints` but **not** replayable.

### Resolved: replay is bit-exact on CUDA; the earlier CPU gap was a harness bug (2026-09-11)

**A CPU replay of step 87,196 initially looked broken.** The loss matched
almost exactly (`ntp` 4.3384 replayed vs 4.3385 captured, 0.002%), but every
per-group gradient norm came out two to three orders of magnitude too
small — an ordinary step, not the spike:

| group | captured | CPU replay | ratio |
|---|---|---|---|
| `override:register` | 2169.2 | 1.0 | 2169x |
| `override:depth_code` | 927.8 | 1.3 | 714x |
| `override:creation_gate` | 902.1 | 1.9 | 475x |
| `V_theta` | 252.6 | 0.5 | 505x |
| `override:log_tau` | 36.1 | 0.5 | 72x |

The ratios were not constant, ruling out a uniform rescale (e.g. a leftover
AMP `GradScaler` factor). Two hypotheses were checked in order:

1. **Gumbel-router RNG stream** — `model_parf_sparse.py`'s top-k router
   draws noise via `torch.rand_like(pi)`, which samples from the *device's*
   own generator, so a CPU replay of a CUDA capture necessarily consumes a
   different stream. **Measured and refuted**: varying the seed on one
   microbatch moved gradients by roughly 25%, not 2000x.
2. **Device/precision execution** (TF32, reduction order, a
   marginally-stable computation near the `omega*dt` stability wall) — the
   remaining candidate. **Refuted by running `replay_spike_batch` on an
   actual A100** and comparing against the same bundles:

| step | fidelity_gap_pct on CUDA | captured `top_groups`, replayed |
|---|---|---|
| 87196 | **0.0002%** | bit-exact: 2169.18, 927.84, 902.07, 549.68, 252.61, 147.66, 36.06, 32.89 |
| 90360 | **0.0003%** | bit-exact: 490.75, 405.74, 338.65, 134.56, 122.93, 109.34, 84.86, 42.26 |

Both match the notebook's long-standing ~0.0019%-fidelity claim for this era
of capture (Diagnostic Programme §7/§13-§15). A computation genuinely
sensitive to device/precision noise near a stability wall could not
reproduce to five nines like this, so the device hypothesis is dead, not
merely unconfirmed — **the spikes are fully deterministic given the correct
model, weights, batch, and RNG state.**

**The real cause of the CPU gap was a bug in the throwaway CPU harness**
(`build_model.py` / `replay_fidelity.py`, session-scratch scripts, never
part of this package) — not in `semsimula_diag`, and not a property of the
model. Its own correctness stands on the parity tests against
`grad_clip_utils.py`, which are independent of any of this. The harness bug
itself was not root-caused and is not being chased further — it does not
block anything, since the real workflow always replays on CUDA.

Two *other* real bugs surfaced while getting the CUDA run working, both
worth keeping since they will recur for anyone calling `replay_spike_batch`
from a fresh session without having run Cell 6:

1. **Bundle batches are numpy arrays, not tensors.** `torch.as_tensor(...)`
   is needed before the forward pass, or `nn.Embedding` raises
   `argument 'indices' must be Tensor, not numpy.ndarray`. Pinned as a
   regression test in `test_real_bundles.py`.
2. **`replay_spike_batch` depends on more Cell-6-only globals than
   `forward_with_vreg`.** An AST scan of its free variables found
   `_GRAD_CLIP_CFG`, `GRAD_CLIP_OVERRIDES`, `WATCHDOG_EXCLUDE_GROUPS`, and —
   the one that actually produced a visible symptom —
   `CLIP_THEN_SUM_GROUPS` / `PER_GROUP_CLIP`. `_cts_group_params` reads
   those via `globals().get(...)`, which returns `None` instead of raising,
   so a missing definition silently falls back to `sum_then_clip` instead
   of erroring. That reintroduced, symptom-for-symptom, the exact bug §47
   documents `clip_then_sum` as fixing: `E`/`P` accumulated raw and
   unclipped across all 4 microbatches (~1380 each) instead of being
   clipped per-microbatch to <=32.89 — a brand-new pair of top groups never
   in the original capture, and a 26-42% aggregate fidelity gap, while
   every *other* group was already bit-exact. Worth carrying into the
   `probes` port: raise on a missing required configuration input, don't
   silently no-op.

With the CUDA path confirmed end to end, the two real bundles now give
`test_real_bundles.py` bit-exact reference values (the tables above) that
could anchor a CUDA-only regression job if one is set up later — CPU
coverage stays limited to config correctness, bundle handling, and the
ported clipping code's parity, which is what it was always going to cover.

### Probes: validated against the real step-87196 bundle

`ProbeContext` replaces the notebook's module globals, and **raises rather
than silently no-opping** when a probe's required configuration is absent --
the direct lesson from the `CLIP_THEN_SUM_GROUPS` round-trip above.
`MissingContextError` names every missing field at once and says what each
is needed for.

The stiffness and tokens families were run on the real step-87196 weights
(CPU, `strict=True` load of all 152 tensors) as well as unit-tested against
a stub `V_theta` with a closed-form spectrum:

| metric | value at step 87196 | reading |
|---|---|---|
| `fro_p50` | **0.99999997** | the Frobenius cap is binding essentially exactly at `sqrt(precision_lr_max)=1.0`, so rank is a pure redistribution knob |
| `pr_p50` | **3.678** of rank 4 | 92% of the maximum participation ratio -- the rank-4 budget is close to saturated |
| spectrum | 1.0, 0.838, 0.726, 0.642 | nearly flat; no collapse onto one dominant direction |
| `sigma_max_sq_p50` | 0.360 | agrees between `sigma_lr_report` and `sigma_lr_spectrum_report` |

Read against the Post-100K checklist's §2.1 decision rule (`pr_p50 >= 3.0`
=> the budget is saturated and rank 8 has a real case), this is a
**saturated** reading. Caveat: it is measured at a *spike* checkpoint, not
at `_best.pt`, which is what §2.1 actually specifies -- so treat it as a
strong indication, not the decision.

```bash
pip install -e ".[test]"
pytest -q
```
