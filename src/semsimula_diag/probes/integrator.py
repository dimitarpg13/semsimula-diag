"""Is the integrator itself what blows up?

Ported from Cell 6d's ``replay_integrator_ablation`` (Diagnostic Programme
SS13, Mitigations SS40).

Replays one capture under different ``cfg.integrator`` settings --
``baoab_cfc`` (diagonal channel integrated exactly, low-rank as an explicit
kick with an ``omega*dt < 2`` wall) versus ``baoab_cfc_lowrank`` (low-rank
folded into the exact solve). If a spike survives every integrator it is the
loss geometry, not the discretisation.

This is the most expensive probe here: a full replay per arm.
"""
from __future__ import annotations

import contextlib
from typing import Dict, FrozenSet, Iterator, Optional, Sequence

from ..clipping import per_group_grad_norms
from ..report import ProbeResult
from ._engine import replayed, restored_model_state
from .context import ProbeContext

__all__ = ["replay_integrator_ablation"]


@contextlib.contextmanager
def _integrator(model, name: str, analytic: bool = True,
                lowrank_layers: Optional[FrozenSet[int]] = None) -> Iterator[None]:
    saved = (model.cfg.integrator, model.cfg.vtheta_analytic_force,
             getattr(model.cfg, "lowrank_layers", None))
    try:
        model.cfg.integrator = name
        model.cfg.vtheta_analytic_force = analytic
        if lowrank_layers is not None and hasattr(model.cfg, "lowrank_layers"):
            model.cfg.lowrank_layers = lowrank_layers
        yield
    finally:
        (model.cfg.integrator, model.cfg.vtheta_analytic_force,
         _ll) = saved
        if _ll is not None and hasattr(model.cfg, "lowrank_layers"):
            model.cfg.lowrank_layers = _ll


def replay_integrator_ablation(
    ctx: ProbeContext, step_tag: int,
    integrators: Sequence[str] = ("baoab_cfc", "baoab_cfc_lowrank"),
    lowrank_layers: Optional[FrozenSet[int]] = None,
    verbose: bool = True,
) -> Dict[str, ProbeResult]:
    """Replay a capture under each integrator in turn.

    Args:
        lowrank_layers: restrict the low-rank exact solve to these layer
            indices (e.g. ``frozenset({0, 1, 2})``), when the model supports
            it -- the usual use is to fold in only the layers carrying
            meaningful salience.

    Include the run's LIVE integrator as a fidelity check: that arm should
    reproduce the recorded ``pre_clip_grad_norm``.
    """
    ctx.require("store", "clip_cfg", "forward_fn")
    bundle, path = ctx.store.load(step_tag)
    if verbose:
        print(f'[integabl] loaded {path.name}  step={bundle["step"]}  '
              f'pre_clip_grad_norm={bundle.get("pre_clip_grad_norm")}')

    out: Dict[str, ProbeResult] = {}
    with restored_model_state(ctx, grads=False, weights=False, rng=False):
        for name in integrators:
            with _integrator(ctx.model, name, lowrank_layers=lowrank_layers):
                with replayed(ctx, bundle, per_layer=True) as info:
                    pg = per_group_grad_norms(ctx.model, ctx.clip_cfg)
                    excl = ctx.clip_cfg.watchdog_exclude_groups
                    total = sum(v * v for k, v in pg.items()
                                if k not in excl) ** 0.5
                    gap = info.fidelity_gap_pct(ctx)
                    per_layer = dict(sorted(info.per_layer_h_grad.items()))
                    ntp = info.ntp
            out[name] = ProbeResult(
                probe_name="integrator", step_tag=bundle.get("step", step_tag),
                fidelity_gap_pct=gap,
                metrics={"pre_clip_grad_norm": total, "ntp": ntp},
                per_layer=per_layer or None, per_group=pg)
            if verbose:
                print(f"  {name:24s} total={total:10.2f}  ntp={ntp:.4f}"
                      + (f"  (fidelity {gap:.4f}%)" if gap is not None else ""))
    return out
