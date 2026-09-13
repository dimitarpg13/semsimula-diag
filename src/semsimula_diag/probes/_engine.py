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
import contextvars
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from ..clipping import ClipThenSum, per_group_grad_norms
from .context import ProbeContext

__all__ = ["ReplayInfo", "replayed", "restored_model_state",
           "patched_attrs", "assert_unpatched", "iter_isolated_rows",
           "CURRENT_MICROBATCH"]


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
    # per_layer_h_grad keeps only the FIRST microbatch (setdefault), which is
    # wrong whenever a spike lives in a later one -- at step 87196 microbatch 2
    # carried 99.94% of the gradient and microbatch 0 carried 0.05%, so a
    # profile read from it described an innocent pass. This keeps every one.
    per_layer_by_mb: Dict[int, Dict[int, float]] = field(default_factory=dict)
    microbatch: int = 0
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


# The microbatch `replayed` is currently on. Hooks installed by other probes
# (notably probes.resonance) need it and cannot reconstruct it: under gradient
# checkpointing the layer step is re-entered on recompute, so counting layer
# wraparounds over-counts badly -- measured 3 calls per (microbatch, layer) on
# the deployed model, which inferred 12 microbatches where there were 4.
CURRENT_MICROBATCH: "contextvars.ContextVar[int]" = contextvars.ContextVar(
    "semsimula_diag_microbatch", default=0)


_ACTIVE_PATCHES = "_semsimula_diag_active_patches"


@contextlib.contextmanager
def patched_attrs(obj: Any,
                  patches: Dict[str, Callable[[Callable], Callable]]
                  ) -> Iterator[None]:
    """Temporarily replace several attributes on ``obj``, guaranteeing
    restoration -- ``restored_model_state`` for arbitrary monkeypatches
    rather than model weights/grads/RNG.

    ``patches`` maps attribute name -> ``make_wrapper(original) ->
    replacement``. Generalises the single-attribute swap
    ``probes.stiffness`` uses internally to cover probes that need to hook
    MULTIPLE call sites at once -- e.g.
    :func:`~semsimula_diag.probes.tau_saturation.probe_hot_rows` patches two
    module-level readout functions simultaneously to record their inputs
    before delegating to the original.

    Every named attribute must already exist on ``obj``; a typo or a wrong
    module produces an ``AttributeError`` from ``getattr`` immediately,
    rather than a hook that silently never fires.

    Which names are currently patched is recorded on ``obj`` itself, so
    :func:`assert_unpatched` can spot a patch that outlived its ``with``
    block. The record lives on the object rather than on the wrapper
    because a ``make_wrapper`` is free to return a bound method or any
    other callable that rejects attribute assignment.
    """
    originals = {name: getattr(obj, name) for name in patches}
    active = getattr(obj, _ACTIVE_PATCHES, None)
    if active is None:
        active = set()
        setattr(obj, _ACTIVE_PATCHES, active)
    try:
        for name, make_wrapper in patches.items():
            setattr(obj, name, make_wrapper(originals[name]))
            active.add(name)
        yield
    finally:
        for name, original in originals.items():
            setattr(obj, name, original)
            active.discard(name)


def assert_unpatched(obj: Any, *names: str) -> None:
    """Raise if any of ``names`` on ``obj`` is still a :func:`patched_attrs`
    wrapper left behind by an earlier, unfinished probe.

    ``patched_attrs`` is a generator-based context manager, so a
    ``KeyboardInterrupt`` -- a user stopping a long replay in a notebook --
    can orphan it mid-``yield`` without running its ``finally``. The
    restore then fires whenever CPython garbage-collects the abandoned
    generator: an arbitrary later moment, possibly in the middle of an
    unrelated forward/backward.

    Both outcomes are bad and only one of them is loud. If the collection
    lands between a gradient-checkpointed forward and its backward
    recompute, the two disagree about which tensors were saved and
    ``torch.utils.checkpoint`` raises ``CheckpointError``. If it lands
    anywhere else, the leaked wrapper just keeps firing -- an ablation's
    *untruncated reference* arm silently runs truncated, and since every
    ``relative_force_error`` is measured against that reference, the whole
    sweep is quietly wrong while looking entirely plausible.

    Call this at the top of any probe that patches, so a poisoned kernel is
    reported as one instead of returning numbers.
    """
    active = getattr(obj, _ACTIVE_PATCHES, None) or set()
    stale = [n for n in names if n in active]
    if stale:
        raise RuntimeError(
            f"{type(obj).__name__}.{', '.join(stale)} still carries a "
            "semsimula_diag patch from an earlier probe that did not finish "
            "cleanly -- most likely an interrupted replay. Results from this "
            "state cannot be trusted: an untruncated reference arm would run "
            "truncated. Restart the kernel and re-run from the setup cells. "
            "Note that `git pull` alone will NOT pick up new code, since the "
            "module is already imported.")


def iter_isolated_rows(
    ctx: ProbeContext, bundle: Dict[str, Any]
) -> Iterator[Tuple[int, int, torch.Tensor, torch.Tensor, int]]:
    """Yield ``(microbatch_idx, row_idx, x_row, y_row, n_rows_in_microbatch)``
    for every row in a capture, one at a time.

    Before each row: the RNG stream is reset to the bundle's pinned state and
    every parameter's ``.grad`` is cleared. Both matter -- without the RNG
    reset, row *i*'s routing depends on how many rows already ran (the
    stream would have been consumed by them); without the grad clear, a
    row's ``.backward()`` would accumulate onto the previous row's gradient
    instead of isolating its own.

    This is the shared primitive behind
    :func:`~semsimula_diag.probes.row_attribution.attribute_spike_rows` and
    :func:`~semsimula_diag.probes.tau_saturation.probe_hot_rows`.

    The caller is responsible for having loaded the bundle's weights first
    (typically by running this *inside* an active
    :func:`replayed` block, whose restoration has not fired yet) -- this
    only handles the per-row RNG/grad reset, not the weight load itself.
    """
    model = ctx.model
    for mb, (xb, yb) in enumerate(bundle["batches"]):
        n_rows = xb.shape[0]
        for row in range(n_rows):
            torch.set_rng_state(bundle["rng_state_cpu"])
            if (bundle.get("rng_state_cuda") is not None
                    and ctx.device == "cuda" and torch.cuda.is_available()):
                torch.cuda.set_rng_state_all(bundle["rng_state_cuda"])
            for p in model.parameters():
                p.grad = None
            x = torch.as_tensor(xb[row:row + 1]).long().to(ctx.device)
            y = torch.as_tensor(yb[row:row + 1]).long().to(ctx.device)
            yield mb, row, x, y, n_rows


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
                n = float(g.detach().norm())
                info.per_layer_h_grad.setdefault(li, n)
                info.per_layer_by_mb.setdefault(info.microbatch, {}).setdefault(li, n)
            h_new.register_hook(_hook)
        return out

    model._fock_layer_step = _instrumented
    return lambda: setattr(model, "_fock_layer_step", original)


def _drop_stale_graph_refs(model) -> None:
    """Clear model-side buffers that hold live autograd-graph references.

    ``model_fock_parf_multixi`` accumulates per-layer register-repulsion
    terms on ``_repulsion_terms`` during a training forward, to be drained
    by ``pop_repulsion_loss()`` before ``backward()``; its own docstring
    notes that "the terms hold live graph references".

    A replay that aborts between those two points -- an exception in the
    forward, or in a gradient-checkpoint recompute during the backward --
    leaves that list populated. Two consequences, and the second is the
    one that bites: the pinned graph keeps its whole activation set alive
    in GPU memory, and a later replay that drains the buffer can walk into
    a graph built before the intervening ``load_state_dict`` calls bumped
    every parameter's version counter, which surfaces as

        RuntimeError: one of the variables needed for gradient computation
        has been modified by an inplace operation ... is at version N;
        expected version N-2

    where the gap of 2 is exactly one probe arm's two in-place parameter
    writes (the bundle load, and ``restored_model_state``'s restore).

    Clearing at the top of a replay is safe: nothing valid can be in the
    buffer before the first forward of this replay has run.
    """
    if getattr(model, "_repulsion_terms", None):
        model._repulsion_terms = []


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
            _drop_stale_graph_refs(model)

            cts = ClipThenSum(model, ctx.clip_cfg)
            info.clip_then_sum_groups = sorted(cts.params)

            for i, (xb, yb) in enumerate(bundle["batches"]):
                info.microbatch = i
                CURRENT_MICROBATCH.set(i)
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
            _drop_stale_graph_refs(model)
            if restore_layer is not None:
                restore_layer()
