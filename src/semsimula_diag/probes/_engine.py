"""The shared deterministic-replay engine every replay probe sits on.

In the notebook each ``replay_*`` helper re-implements the same sequence:
snapshot grads/weights/RNG, load the bundle's pinned pre-step weights,
restore the RNG, run the captured microbatches through the training loss with
clip_then_sum applied per microbatch, then restore everything in a
``finally``. Getting any step of that subtly wrong changes the answer without
raising -- which is exactly what happened on real hardware when
``clip_then_sum`` silently degraded to ``sum_then_clip`` and inflated two
groups by three orders of magnitude.

Centralising it means the sequence is written (and tested) once.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch

from ..clipping import ClipThenSum, per_group_grad_norms
from .context import ProbeContext

__all__ = ["ReplayInfo", "replayed", "restored_model_state"]


@dataclass
class ReplayInfo:
    """What the replay observed, available inside the ``replayed`` block."""

    step: Optional[int] = None
    grad_accum: int = 1
    recorded_pre_clip: Optional[float] = None
    ntp: float = 0.0
    v_reg: float = 0.0
    fock_reg: float = 0.0
    per_layer_h_grad: Dict[int, float] = field(default_factory=dict)
    clip_then_sum_groups: List[str] = field(default_factory=list)

    def replayed_total(self, ctx: ProbeContext) -> float:
        """Aggregate pre-clip norm over non-excluded groups, matching what
        the live watchdog compared against ``hard_trigger``."""
        ctx.require("clip_cfg")
        excl = ctx.clip_cfg.watchdog_exclude_groups
        pg = per_group_grad_norms(ctx.model, ctx.clip_cfg)
        return sum(v * v for k, v in pg.items() if k not in excl) ** 0.5

    def fidelity_gap_pct(self, ctx: ProbeContext) -> Optional[float]:
        """Percent disagreement between the replay and what training
        recorded. The notebook treats gaps up to ~0.002% as bit-exact and
        warns above 5%."""
        if not self.recorded_pre_clip:
            return None
        got = self.replayed_total(ctx)
        return 100.0 * abs(got - self.recorded_pre_clip) / max(
            self.recorded_pre_clip, 1e-9)


@contextlib.contextmanager
def restored_model_state(ctx: ProbeContext, *, grads: bool = True,
                         weights: bool = True, rng: bool = True) -> Iterator[None]:
    """Restore whatever the block disturbs -- the non-pollution invariant.

    Safe to wrap around a probe that runs against a model which will keep
    training afterwards.
    """
    model = ctx.model
    saved_grads = ({n: (p.grad.detach().clone() if p.grad is not None else None)
                    for n, p in model.named_parameters()} if grads else None)
    saved_sd = copy.deepcopy(model.state_dict()) if weights else None
    saved_rng_cpu = torch.get_rng_state() if rng else None
    saved_rng_cuda = (torch.cuda.get_rng_state_all()
                      if rng and ctx.device == "cuda" and torch.cuda.is_available()
                      else None)
    was_training = model.training
    try:
        yield
    finally:
        if saved_sd is not None:
            model.load_state_dict(saved_sd)
        if saved_grads is not None:
            for n, p in model.named_parameters():
                g = saved_grads.get(n)
                p.grad = g.clone() if g is not None else None
        if saved_rng_cpu is not None:
            torch.set_rng_state(saved_rng_cpu)
        if saved_rng_cuda is not None:
            torch.cuda.set_rng_state_all(saved_rng_cuda)
        model.train(was_training)


def _install_layer_hook(ctx: ProbeContext, info: ReplayInfo) -> Optional[Callable]:
    """Record the gradient flowing into each layer boundary.

    Returns a restore callable, or None when the model exposes no
    ``_fock_layer_step`` to wrap.
    """
    model = ctx.model
    original = getattr(model, "_fock_layer_step", None)
    if original is None:
        return None

    def _instrumented(h, h_prev, r, salience, m_b, gamma, dt, layer_idx,
                      *args, **kwargs):
        out = original(h, h_prev, r, salience, m_b, gamma, dt, layer_idx,
                       *args, **kwargs)
        h_new = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(h_new) and h_new.requires_grad:
            def _hook(g, li=layer_idx):
                # MUST NOT return the setdefault() result: torch treats any
                # non-None tensor-hook return as a gradient replacement, and
                # setdefault returns a float -> "expected Variable, but hook
                # returned 'float'".
                info.per_layer_h_grad.setdefault(li, float(g.detach().norm()))
            h_new.register_hook(_hook)
        return out

    model._fock_layer_step = _instrumented
    return lambda: setattr(model, "_fock_layer_step", original)


@contextlib.contextmanager
def replayed(ctx: ProbeContext, bundle: Dict[str, Any], *,
             load_weights: bool = True,
             per_layer: bool = False,
             on_microbatch: Optional[Callable[[int], None]] = None,
             ) -> Iterator[ReplayInfo]:
    """Re-run a capture's microbatches, leaving the resulting ``.grad`` in
    place for the caller to measure, then restore everything.

    Inside the block, ``ctx.model``'s gradients are exactly what the live
    training step produced -- assuming the context is configured to match.

    Args:
        load_weights: load the bundle's pinned pre-step weights. Set False
            to replay the captured batch against the *current* weights.
        per_layer: also record per-layer boundary gradients (needs the model
            to expose ``_fock_layer_step``).
        on_microbatch: called with each microbatch index after its backward
            and clip-then-sum fold -- for probes that need per-microbatch
            state before the next one overwrites it.

    Raises:
        MissingContextError: if ``forward_fn``, ``clip_cfg`` or ``grad_accum``
            is absent. It never silently degrades to a partial replay.
    """
    ctx.require("forward_fn", "clip_cfg")
    model = ctx.model
    info = ReplayInfo(
        step=bundle.get("step"),
        grad_accum=ctx.grad_accum or bundle.get("grad_accum",
                                                len(bundle["batches"])),
        recorded_pre_clip=bundle.get("pre_clip_grad_norm"),
    )

    with restored_model_state(ctx):
        restore_layer = _install_layer_hook(ctx, info) if per_layer else None
        try:
            if load_weights:
                model.load_state_dict(
                    {k: v.to(ctx.device)
                     for k, v in bundle["model_state_dict"].items()},
                    strict=False)
            # RNG must be restored AFTER the weight load and BEFORE the first
            # forward: the router consumes the stream during the forward pass.
            torch.set_rng_state(bundle["rng_state_cpu"])
            if (bundle.get("rng_state_cuda") is not None
                    and ctx.device == "cuda" and torch.cuda.is_available()):
                torch.cuda.set_rng_state_all(bundle["rng_state_cuda"])

            for p in model.parameters():
                p.grad = None
            model.train()

            cts = ClipThenSum(model, ctx.clip_cfg)
            info.clip_then_sum_groups = sorted(cts.params)

            for i, (xb, yb) in enumerate(bundle["batches"]):
                # bundles store numpy arrays, not tensors
                x = torch.as_tensor(xb).long().to(ctx.device)
                y = torch.as_tensor(yb).long().to(ctx.device)
                if getattr(model, "_fock_capture", None) is not None:
                    model._fock_capture = []     # drain the previous microbatch
                loss, loss_ntp, v_reg, fock_reg = ctx.forward_fn(model, x, y, ctx)
                if ctx.register_repulsion and hasattr(model, "pop_repulsion_loss"):
                    loss = loss + model.pop_repulsion_loss()
                (loss / info.grad_accum).backward()
                cts.apply_microbatch()
                info.ntp += float(loss_ntp) / info.grad_accum
                info.v_reg += float(v_reg) / info.grad_accum
                info.fock_reg += float(fock_reg) / info.grad_accum
                if on_microbatch is not None:
                    on_microbatch(i)

            cts.splice_back()
            yield info
        finally:
            if restore_layer is not None:
                restore_layer()
