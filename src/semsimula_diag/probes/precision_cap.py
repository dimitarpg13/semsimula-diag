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

from ..clipping import per_group_grad_norms
from ..report import ProbeResult
from ._engine import replayed, restored_model_state
from .context import ProbeContext

__all__ = ["replay_precision_cap_ablation", "replay_curvature_rebalance_ablation"]


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
