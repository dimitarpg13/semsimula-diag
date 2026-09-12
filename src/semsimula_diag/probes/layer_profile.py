"""Where in the model a spike's gradient lives.

Ported from Cell 6d's ``replay_spike_batch`` (Diagnostic Programme SS7.1).

**Scope note.** The notebook's version also installs extensive
forward-activation instrumentation (creation-gate alpha/tau extremes,
reverse-channel Q_force ranges, destruction-gate outputs, per-bank V_theta
exponent histograms, register/gate health via ``set_fock_capture``). Those
are additional diagnostics layered on the same replay; what is ported here is
the measurement core the fidelity check and every attribution argument rest
on -- per-parameter norms, per-group norms, the per-layer boundary gradient,
and the fidelity gap. The activation instrumentation is not ported yet.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..clipping import per_group_grad_norms
from ..report import ProbeResult
from ._engine import replayed
from .context import ProbeContext

__all__ = ["replay_spike_batch"]


def replay_spike_batch(ctx: ProbeContext, step_tag: int, top_k: int = 12,
                       verbose: bool = True) -> ProbeResult:
    """Deterministically replay a capture and report where the gradient sat.

    **Read ``fidelity_gap_pct`` first.** It compares the replayed aggregate
    against what training recorded in the bundle. Up to ~0.002% is bit-exact;
    above ~5% the replay is not reproducing the step and every attribution
    below it is unsafe to trust. The usual cause is a context that does not
    match the live run -- most often clip-then-sum configuration.

    Use it when you have a spike and want to know *which* parameters and
    *which* layers carried it, before reaching for a more specific probe.
    """
    ctx.require("store", "clip_cfg", "forward_fn")
    bundle, path = ctx.store.load(step_tag)
    if verbose:
        print(f'[replay] loaded {path.name}  step={bundle["step"]}  '
              f'pre_clip_grad_norm={bundle.get("pre_clip_grad_norm")}')
        if bundle.get("top_groups"):
            print("[replay] Phase-0 top groups at capture time: "
                  + ", ".join(f"{k}={v}" for k, v in bundle["top_groups"].items()))

    with replayed(ctx, bundle, per_layer=True) as info:
        param_norms = {n: float(p.grad.detach().norm())
                       for n, p in ctx.model.named_parameters()
                       if p.grad is not None}
        group_norms = per_group_grad_norms(ctx.model, ctx.clip_cfg)
        excl = ctx.clip_cfg.watchdog_exclude_groups
        total_excl = sum(v * v for k, v in group_norms.items()
                         if k not in excl) ** 0.5
        total_all = sum(v * v for v in group_norms.values()) ** 0.5
        gap = info.fidelity_gap_pct(ctx)
        per_layer = dict(sorted(info.per_layer_h_grad.items()))
        top_params = sorted(param_norms.items(), key=lambda kv: -kv[1])[:top_k]

    if verbose:
        print(f"\n[replay] fidelity check: replayed (matching-groups) total="
              f"{total_excl:.1f}  vs. recorded="
              f"{info.recorded_pre_clip}  (diff {gap:.4f}%)"
              if gap is not None else "")
        if gap is not None and gap > 5.0:
            print("[replay][WARN] fidelity gap > 5% -- the replay is not "
                  "reproducing the step. Check that clip_cfg's "
                  "clip_then_sum_groups/threshold match the live run before "
                  "trusting any attribution below.")
        print(f"[replay] replayed total incl. excluded groups={total_all:.1f} "
              f"(excl.={total_excl:.1f}; the gap is what the watchdog missed)")
        print("[replay] replayed per-group norms (top 8):")
        for k, v in sorted(group_norms.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {v:10.2f}  {k}")
        print(f"[replay] top {top_k} per-parameter grad norms:")
        for n, v in top_params:
            print(f"    {v:10.2f}  {n}")
        if per_layer:
            print("[replay] per-layer grad norm (into each layer boundary):")
            for li, v in per_layer.items():
                print(f"    layer {li:2d}: {v:10.2f}")

    return ProbeResult(
        probe_name="layer_profile",
        step_tag=bundle.get("step", step_tag),
        fidelity_gap_pct=gap,
        metrics={
            "pre_clip_grad_norm_recorded": info.recorded_pre_clip or 0.0,
            "pre_clip_grad_norm_replayed_matching": total_excl,
            "pre_clip_grad_norm_replayed_all_groups": total_all,
            "ntp": info.ntp, "v_reg": info.v_reg, "fock_reg": info.fock_reg,
        },
        per_layer=per_layer or None,
        per_group=group_norms,
        raw={"top_params": top_params,
             "clip_then_sum_groups": info.clip_then_sum_groups},
    )
