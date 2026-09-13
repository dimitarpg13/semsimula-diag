"""Does capping the low-rank curvature channel tame a spike?

Ported from Cell 6d's ``replay_precision_cap_ablation`` and
``replay_curvature_rebalance_ablation`` (Diagnostic Programme SS17,
Mitigations SS41.7/SS42).

**Why this one is a genuine ablation and clip thresholds are not.**
``precision_lr_max`` changes the FORWARD pass -- ``_bound_lowrank`` reads
``self._precision_lr_max`` fresh on every forward -- so replaying the same
batch under a different budget produces genuinely different gradients. A clip
threshold, by contrast, is a pure post-hoc rescale of an already-computed
gradient, so it cannot be "ablated" by replay at all (see
``probes.clip_order`` for what *can* be tested there).
"""
from __future__ import annotations

import contextlib
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import torch

from ..clipping import per_group_grad_norms
from ..report import ProbeResult
from ._engine import (assert_unpatched, patched_attrs, replayed,
                      restored_model_state)
from .context import ProbeContext

__all__ = ["replay_precision_cap_ablation", "replay_curvature_rebalance_ablation",
          "replay_rank_truncation_ablation"]


def _banks(model) -> List:
    bank = getattr(model.V_theta, "banks", None)
    if bank is None:
        bank = getattr(getattr(model.V_theta, "bank", None), "banks", None)
    if bank is None:
        raise RuntimeError(
            "V_theta exposes no .banks; need the anisotropic Gaussian family.")
    return list(bank)


@contextlib.contextmanager
def _budgets(model, lr_max=..., diag_max=...) -> Iterator[None]:
    """Temporarily set every bank's caps, restoring both afterwards."""
    banks = _banks(model)
    saved_lr = [b._precision_lr_max for b in banks]
    saved_diag = [getattr(b, "_precision_max", None) for b in banks]
    try:
        for b in banks:
            if lr_max is not ...:
                b._precision_lr_max = lr_max
            if diag_max is not ... and saved_diag[0] is not None:
                b._precision_max = diag_max
        yield
    finally:
        for b, lr, dg in zip(banks, saved_lr, saved_diag):
            b._precision_lr_max = lr
            if dg is not None:
                b._precision_max = dg


def _measure(ctx: ProbeContext, bundle) -> Dict[str, float]:
    with replayed(ctx, bundle, per_layer=True) as info:
        pg = per_group_grad_norms(ctx.model, ctx.clip_cfg)
        excl = ctx.clip_cfg.watchdog_exclude_groups
        total = sum(v * v for k, v in pg.items() if k not in excl) ** 0.5
        return {"pre_clip_grad_norm": total, "ntp": info.ntp,
                "per_group": pg, "per_layer": dict(sorted(
                    info.per_layer_h_grad.items()))}


def replay_precision_cap_ablation(
    ctx: ProbeContext, step_tag: int,
    budgets: Sequence[Optional[float]] = (1.0, 4.0, None),
    verbose: bool = True,
) -> Dict[str, ProbeResult]:
    """Replay one capture under several ``precision_lr_max`` budgets.

    Args:
        budgets: floats, or ``None`` to disable the cap entirely. Include the
            run's LIVE value as a fidelity check -- that arm should reproduce
            the bundle's recorded ``pre_clip_grad_norm``, and if it does not,
            nothing else in the sweep is trustworthy.

    Use it when a spike is dominated by ``V_theta`` / curvature groups and you
    want to know whether bounding the low-rank channel would have prevented it.
    """
    ctx.require("store", "clip_cfg", "forward_fn")
    bundle, path = ctx.store.load(step_tag)
    if verbose:
        print(f'[precap] loaded {path.name}  step={bundle["step"]}  '
              f'pre_clip_grad_norm={bundle.get("pre_clip_grad_norm")}')

    out: Dict[str, ProbeResult] = {}
    with restored_model_state(ctx, grads=False, weights=False, rng=False):
        for budget in budgets:
            label = ("uncapped (None)" if budget is None
                     else f"precision_lr_max={budget}")
            with _budgets(ctx.model, lr_max=budget):
                m = _measure(ctx, bundle)
            gap = (100.0 * abs(m["pre_clip_grad_norm"]
                               - bundle["pre_clip_grad_norm"])
                   / max(bundle["pre_clip_grad_norm"], 1e-9)
                   if bundle.get("pre_clip_grad_norm") else None)
            out[label] = ProbeResult(
                probe_name="precision_cap",
                step_tag=bundle.get("step", step_tag),
                fidelity_gap_pct=gap,
                metrics={"budget": float("nan") if budget is None else budget,
                         "pre_clip_grad_norm": m["pre_clip_grad_norm"],
                         "ntp": m["ntp"]},
                per_layer=m["per_layer"] or None, per_group=m["per_group"])
            if verbose:
                print(f'  {label:28s} total={m["pre_clip_grad_norm"]:10.2f} '
                      f'ntp={m["ntp"]:.4f}')
    return out


def replay_curvature_rebalance_ablation(
    ctx: ProbeContext, step_tag: int,
    precision_max: Sequence[float] = (),
    precision_lr_max: Sequence[Optional[float]] = (1.0, 0.25),
    verbose: bool = True,
) -> Dict[str, ProbeResult]:
    """2-D sweep over BOTH caps -- the diagonal/low-rank rebalance question.

    The diagonal cap (``precision_max``, historically ``2/d``) is a
    Verlet-era artifact, while the low-rank cap is the one that has been
    tuned. This asks whether curvature should be *redistributed* between the
    two channels rather than merely bounded in one.

    Args:
        precision_max: diagonal caps to try. Empty = leave it at the model's
            current value and sweep only the low-rank cap.
    """
    ctx.require("store", "clip_cfg", "forward_fn")
    bundle, path = ctx.store.load(step_tag)
    if verbose:
        print(f'[rebal] loaded {path.name}  step={bundle["step"]}')

    diag_grid = list(precision_max) or [...]
    out: Dict[str, ProbeResult] = {}
    with restored_model_state(ctx, grads=False, weights=False, rng=False):
        for dmax in diag_grid:
            for lrmax in precision_lr_max:
                dlab = "diag=current" if dmax is ... else f"diag={dmax:g}"
                llab = "lr=None" if lrmax is None else f"lr={lrmax:g}"
                label = f"{dlab}, {llab}"
                with _budgets(ctx.model, lr_max=lrmax, diag_max=dmax):
                    m = _measure(ctx, bundle)
                out[label] = ProbeResult(
                    probe_name="curvature_rebalance",
                    step_tag=bundle.get("step", step_tag),
                    metrics={"pre_clip_grad_norm": m["pre_clip_grad_norm"],
                             "ntp": m["ntp"]},
                    per_group=m["per_group"])
                if verbose:
                    print(f'  {label:28s} total={m["pre_clip_grad_norm"]:10.2f}')
    return out


def _svd_truncate(B: torch.Tensor, rank: int) -> torch.Tensor:
    """Reconstruct ``B`` at reduced effective rank via SVD, preserving its
    original shape.

    ``B`` has shape ``(..., d, r_full)``. Keeps only the top ``rank``
    singular directions and zeros the rest, so the reconstruction has
    exactly ``B``'s shape but at most ``rank`` nonzero singular values --
    no downstream shape changes needed anywhere else in the model.

    ``rank >= r_full`` is a true no-op (returns ``B`` unchanged, skipping
    the SVD entirely) -- this is what makes a truncation ablation's own
    ``rank = r_full`` arm a built-in fidelity check against the genuinely
    untruncated reference.

    **Why this is a Gram-matrix projection and not a literal SVD.** ``B``
    is ``(K, d, r)`` *per token* (see ``AnisotropicGaussianVTheta.
    context_components``: "``B`` alone is ``K * d * rank`` floats per
    token"), so at the deployed shape a single forward pass would issue
    on the order of ``B*T*K*n_ctx*L`` batched ``d x r`` decompositions.
    That is precisely the cost that makes the ``baoab_cfc_lowrank``
    integrator arm non-viable: ``cfc_baoab.py`` measures it at ~120 s/step
    against ``baoab_cfc``'s ~10-15 s/step and records "the batched
    per-token SVD is the entire extra cost". An ablation must not adopt
    the cost profile of the arm it is diagnosing.

    Two facts make the full SVD avoidable. First, ``d >> r`` (384 vs 4),
    and truncation only needs the *right* singular subspace: with the thin
    SVD ``B = U S V^T``, the rank-``r'`` truncation is exactly
    ``B V_{r'} V_{r'}^T``, since ``V^T V_{r'}`` selects the leading block.
    ``V`` is obtained from the ``r x r`` Gram matrix ``B^T B = V S^2 V^T``
    -- a 4x4 ``eigh`` in place of a 384x4 SVD -- and ``U`` is never formed.

    Second, the projector is a *choice of directions*, not a magnitude, so
    it is computed under ``no_grad`` and the truncation reduces to the
    linear map ``B @ P``. Gradients therefore flow into ``B`` (and on into
    the projection weights that produced it) exactly as for any other
    linear operation, while nothing differentiates through the
    decomposition itself. That removes the backward's
    ``1/(sigma_i^2 - sigma_j^2)`` terms outright -- which matters here
    beyond speed, because zeroing the tail deliberately *manufactures*
    repeated singular values, and ``cfc_baoab.py`` documents that a
    degenerate batch element can make cuSOLVER "return NaN singular
    vectors/values *without raising*", silently poisoning the gradient.

    The forward value is unchanged (still the exact rank-``r'``
    truncation); only the subspace's own sensitivity is held fixed, which
    is the intended reading of the ablation -- project the realised well
    onto its leading directions and ask what the optimizer would then have
    seen.

    **Why the r x r eigh runs on CPU, when the d x r SVD must not.**
    cuSOLVER rejects this batch outright on CUDA 13.0, failing in
    ``cusolverDnXsyevBatched_bufferSize`` -- the workspace-sizing call,
    which runs before any matrix element is read, so despite the generic
    "may appear if the input matrix contains NaN" hint attached to every
    cuSOLVER error, this is parameter validation refusing 32,768 batched
    4x4 problems, not bad data. Routing just the decomposition to CPU
    sidesteps the batched eigensolver entirely.

    This is emphatically not the CPU fallback ``cfc_baoab.py`` warns
    about, for three reasons. That warning concerns decomposing ``d x m``
    matrices, and the Gram reduction above has already shrunk the problem
    to ``r x r``; only the Gram matrix crosses the bus (2.1 MB at the
    deployed shape, against 201 MB for ``B`` itself), with both matmuls
    staying on the accelerator; and the routing here is unconditional,
    whereas that warning is really about a *conditional* fallback whose
    branch can differ between a forward pass and its checkpoint recompute.
    Measured at the deployed shape, the eigh costs ~68 ms per call and
    ~32 s across a four-rank ablation.
    """
    if rank >= B.shape[-1]:
        return B
    with torch.no_grad():
        gram = B.transpose(-2, -1) @ B          # (..., r, r), stays on device
        gram_cpu = gram.cpu()
        if not torch.isfinite(gram_cpu).all():
            raise RuntimeError(
                "non-finite Gram matrix B^T B in rank truncation: the "
                "replayed well parameters are already corrupt before any "
                "truncation is applied, so the ablation would measure "
                "noise. Check the untruncated arm's fidelity first.")
        # eigh gives ascending eigenvalues, so the top `rank` directions
        # are the trailing columns.
        _, evecs = torch.linalg.eigh(gram_cpu)
        v_top = evecs[..., -rank:].to(B.device)
        projector = v_top @ v_top.transpose(-2, -1)
    return B @ projector


def replay_rank_truncation_ablation(
    ctx: ProbeContext, step_tag: int,
    ranks: Sequence[int] = (1, 2, 3, 4),
    verbose: bool = True,
) -> Dict[str, ProbeResult]:
    """Does the well actually use its rank budget, or is truncating it free?

    Ported from the rank-selection procedure's Stage 2 (Curvature
    Diagnostics and Rank Selection for Aniso Gaussian V_theta, SS7.3),
    proposed there and built here for the first time.

    The participation ratio (``probes.stiffness.sigma_lr_spectrum_report``)
    is a purely geometric statistic: it says how the well's curvature
    *budget* is distributed across directions, not whether the small
    directions carry any *force*. A well could have a low participation
    ratio yet still depend functionally on its tail singular vectors, if
    they happen to align with something the loss cares about. This probe
    is the direct test: replay a captured batch with the realised ``B_k``
    truncated to its rank-``r'`` SVD reconstruction, for each ``r'`` in
    ``ranks``, and report how much the resulting gradient actually moves.

    Hooks ``context_components`` -- the single point where the integrator
    computes the realised (post-``_bound_lowrank``) well parameters once
    per layer and reuses them for both the harmonic split and the force
    (see the docstring on ``context_components`` itself) -- so every
    downstream consumer for that step sees the truncated ``B`` uniformly.

    Args:
        ranks: SVD ranks to truncate to. The default follows the note's
            proposed signature; a value ``>=`` the model's actual rank is a
            no-op (see :func:`_svd_truncate`) and only costs the fidelity
            check, not a real ablation.

    Returns a dict keyed ``"full (untruncated)"`` plus one entry per rank in
    ``ranks``, each a :class:`ProbeResult` with ``metrics['rank']``,
    ``metrics['pre_clip_grad_norm']``, ``metrics['ntp']``, and the decisive
    reading, ``metrics['relative_force_error']`` -- ``||g_trunc - g_full|| /
    ||g_full||`` over the full flattened parameter-gradient vector. A
    truncation that leaves this near zero means the truncated directions
    were dead weight; a truncation that moves it substantially means they
    were carrying real signal.

    **Scope note.** "Force" here is the parameter gradient the training
    loss produces through the truncated well, not a separately-captured
    per-position ``h``-space force field -- cheaper to compute, reuses the
    same replay machinery as every other ablation in this module, and
    answers the question the note actually poses: does truncating rank
    change what the optimizer would have done.
    """
    ctx.require("store", "clip_cfg", "forward_fn")
    # A leaked patch would make the untruncated reference arm run truncated,
    # silently rescaling every relative_force_error below.
    assert_unpatched(ctx.model.V_theta, "context_components")
    bundle, path = ctx.store.load(step_tag)
    if verbose:
        print(f'[ranktrunc] loaded {path.name}  step={bundle["step"]}  '
              f'pre_clip_grad_norm={bundle.get("pre_clip_grad_norm")}')

    def _flat_grad(model) -> torch.Tensor:
        return torch.cat([p.grad.detach().flatten()
                          for p in model.parameters() if p.grad is not None])

    def _measure(rank: Optional[int]):
        if rank is None:
            cm: contextlib.AbstractContextManager = contextlib.nullcontext()
        else:
            def _make_wrapper(original):
                def _wrapped(xis):
                    comps = original(xis)
                    return [(mu, a, w, _svd_truncate(B, rank) if B.shape[-1] else B)
                            for (mu, a, w, B) in comps]
                return _wrapped
            cm = patched_attrs(ctx.model.V_theta,
                               {"context_components": _make_wrapper})
        with cm:
            with replayed(ctx, bundle) as info:
                pg = per_group_grad_norms(ctx.model, ctx.clip_cfg)
                excl = ctx.clip_cfg.watchdog_exclude_groups
                total = sum(v * v for k, v in pg.items() if k not in excl) ** 0.5
                grad_vec = _flat_grad(ctx.model)
                ntp = info.ntp
        return total, ntp, grad_vec, pg

    out: Dict[str, ProbeResult] = {}
    with restored_model_state(ctx, grads=False, weights=False, rng=False):
        full_total, full_ntp, full_grad, full_pg = _measure(None)
        full_norm = float(full_grad.norm()) or 1e-12
        out["full (untruncated)"] = ProbeResult(
            probe_name="rank_truncation", step_tag=bundle.get("step", step_tag),
            metrics={"rank": float("inf"), "pre_clip_grad_norm": full_total,
                     "ntp": full_ntp, "relative_force_error": 0.0},
            per_group=full_pg)
        if verbose:
            print(f'  {"full (untruncated)":20s} total={full_total:10.2f}  '
                  f'ntp={full_ntp:.4f}')

        for r in ranks:
            total, ntp, grad_vec, pg = _measure(r)
            rel_err = float((grad_vec - full_grad).norm() / full_norm)
            out[f"rank={r}"] = ProbeResult(
                probe_name="rank_truncation", step_tag=bundle.get("step", step_tag),
                metrics={"rank": r, "pre_clip_grad_norm": total, "ntp": ntp,
                         "relative_force_error": rel_err},
                per_group=pg)
            if verbose:
                print(f'  rank={r:<16d} total={total:10.2f}  ntp={ntp:.4f}  '
                      f'relative_force_error={rel_err:.4f}')
    return out
