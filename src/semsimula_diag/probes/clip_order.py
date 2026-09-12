"""Does it matter *when* clipping is applied, not just how tight it is?

Ported from Cell 6d's ``replay_clip_ablation`` (Mitigations SS45.3).

**The correctness point this probe exists to make concrete.**
``clip_grads_per_group`` is a pure post-hoc rescale:
``nn.utils.clip_grad_norm_`` returns the norm computed BEFORE it rescales, so
the reported pre-clip norm is identical no matter what threshold you pass.
A clip threshold therefore cannot be ablated by replaying the same
accumulated gradient -- ``min(threshold, raw_norm)`` is already known once
``raw_norm`` is.

What genuinely differs is WHERE in the accumulation the clip sits:

- ``sum_then_clip`` -- accumulate raw across every microbatch, clip the total
  once. One outlier microbatch can dominate the sum before the clip sees it;
  the clip then shrinks the *result* uniformly and cannot stop the outlier
  setting the update's *direction*.
- ``clip_then_sum`` -- clip each microbatch's own contribution first, which
  bounds any single microbatch at the source.

Both arms replay the same captured microbatches and RNG, so the only
difference is order.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn

from ..clipping import GradClipConfig, assign_clip_group
from ..report import ProbeResult
from ._engine import restored_model_state
from .context import ProbeContext

__all__ = ["replay_clip_ablation"]


def _group_params(model, cfg: GradClipConfig, groups) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key, _ = assign_clip_group(n, cfg)
        if key in groups:
            out.setdefault(key, []).append(p)
    return out


def replay_clip_ablation(ctx: ProbeContext, step_tag: int,
                         groups: Sequence[str] = ("E", "P"),
                         thresholds: Sequence[float] = (1.0, 0.3, 0.1, 0.03),
                         verbose: bool = True) -> ProbeResult:
    """Compare ``sum_then_clip`` against ``clip_then_sum`` for ``groups``.

    Reports, per group and threshold, the resulting applied gradient norm
    under each order, plus the cosine between the two resulting update
    directions -- a cosine well below 1 means the orders disagree about
    *where* to step, not merely how far.
    """
    ctx.require("store", "clip_cfg", "forward_fn")
    bundle, path = ctx.store.load(step_tag)
    if verbose:
        print(f'[clipabl] loaded {path.name}  step={bundle["step"]}')

    model = ctx.model
    grad_accum = ctx.grad_accum or bundle.get("grad_accum",
                                              len(bundle["batches"]))
    gset = frozenset(groups)

    def _accumulate(clip_first: bool, threshold: float) -> Dict[str, torch.Tensor]:
        """Return each group's flattened resulting gradient."""
        gparams = _group_params(model, ctx.clip_cfg, gset)
        running: Dict[int, torch.Tensor] = {}
        for p in model.parameters():
            p.grad = None
        torch.set_rng_state(bundle["rng_state_cpu"])
        if (bundle.get("rng_state_cuda") is not None
                and ctx.device == "cuda" and torch.cuda.is_available()):
            torch.cuda.set_rng_state_all(bundle["rng_state_cuda"])
        for xb, yb in bundle["batches"]:
            x = torch.as_tensor(xb).long().to(ctx.device)
            y = torch.as_tensor(yb).long().to(ctx.device)
            loss, *_ = ctx.forward_fn(model, x, y, ctx)
            if ctx.register_repulsion and hasattr(model, "pop_repulsion_loss"):
                loss = loss + model.pop_repulsion_loss()
            (loss / grad_accum).backward()
            if clip_first:
                for ps in gparams.values():
                    if any(p.grad is not None for p in ps):
                        nn.utils.clip_grad_norm_(ps, threshold)
                        for p in ps:
                            if p.grad is None:
                                continue
                            pid = id(p)
                            c = p.grad.detach().clone()
                            running[pid] = (running[pid] + c if pid in running
                                            else c)
                            p.grad.zero_()
        out: Dict[str, torch.Tensor] = {}
        for key, ps in gparams.items():
            if clip_first:
                for p in ps:
                    if id(p) in running:
                        p.grad = running[id(p)]
            else:
                nn.utils.clip_grad_norm_(ps, threshold)
            out[key] = torch.cat([p.grad.detach().flatten() for p in ps
                                  if p.grad is not None])
        return out

    sts: Dict[str, Dict[float, float]] = {g: {} for g in gset}
    cts: Dict[str, Dict[float, float]] = {g: {} for g in gset}
    cos: Dict[str, Dict[float, float]] = {g: {} for g in gset}

    with restored_model_state(ctx):
        model.load_state_dict(
            {k: v.to(ctx.device) for k, v in bundle["model_state_dict"].items()},
            strict=False)
        model.train()
        for thr in thresholds:
            a = _accumulate(False, thr)
            b = _accumulate(True, thr)
            for g in a:
                sts[g][thr] = float(a[g].norm())
                cts[g][thr] = float(b[g].norm())
                cos[g][thr] = float(torch.nn.functional.cosine_similarity(
                    a[g].unsqueeze(0), b[g].unsqueeze(0)).item())

    if verbose:
        for g in sorted(sts):
            print(f"\n  group {g}:")
            print(f'    {"threshold":>10} {"sum_then_clip":>14} '
                  f'{"clip_then_sum":>14} {"cosine":>9}')
            for thr in thresholds:
                print(f"    {thr:10.3f} {sts[g][thr]:14.4f} "
                      f"{cts[g][thr]:14.4f} {cos[g][thr]:9.4f}")
        print("\n  A cosine below 1 means the two orders disagree about the "
              "update DIRECTION, which no threshold can fix.")

    return ProbeResult(
        probe_name="clip_order",
        step_tag=bundle.get("step", step_tag),
        metrics={"n_groups": len(sts), "n_thresholds": len(thresholds)},
        raw={"sum_then_clip": sts, "clip_then_sum": cts,
             "cosine_vs_sum_then_clip": cos},
    )
