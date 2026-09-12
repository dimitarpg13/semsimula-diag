"""What a probe needs from its run, made explicit.

In the notebook every probe reaches into module globals -- `model`, `DEVICE`,
`GRAD_ACCUM`, `_GRAD_CLIP_CFG`, `LAMBDA_V`, `CKPT_DIR`, and so on. That is
what makes Cell 6d impossible to load without Cell 6 having run first, and it
is the single largest source of friction in the port.

:class:`ProbeContext` collects those into one object passed explicitly.

**Missing configuration raises.** The notebook's ``_cts_group_params`` reads
its settings with ``globals().get(...)``, which returns ``None`` rather than
raising, so an unset ``CLIP_THEN_SUM_GROUPS`` silently degraded clip_then_sum
into sum_then_clip -- reintroducing the exact bug Mitigations SS45.3-45.4
exists to fix, and costing a full debugging round-trip on real hardware
before the symptom was recognised. :meth:`ProbeContext.require` makes that
class of mistake loud instead of silent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from ..clipping import GradClipConfig
from ..replay import BundleStore

__all__ = ["ProbeContext", "MissingContextError"]


class MissingContextError(RuntimeError):
    """A probe needed a piece of run configuration that was not supplied."""


@dataclass
class ProbeContext:
    """Everything a probe may need, supplied explicitly rather than read
    from globals.

    Only ``model`` is always required; each probe declares the rest it needs
    via :meth:`require`, so a probe that only reads weights does not force a
    caller to invent a batch provider or a clip config.

    Attributes:
        model: the network under study. Probes restore whatever they mutate.
        device: where to run; ``'cpu'`` is fine for weight-only probes.
        store: bundle lookup (live ring, then archive). Needed by any probe
            that loads a capture by step tag.
        clip_cfg: per-group clip configuration, including the clip_then_sum
            fields. Needed by anything that measures per-group grad norms.
        grad_accum: microbatch count. Defaults to the bundle's own
            ``grad_accum`` when a probe is replaying a capture.
        lambda_v / lambda_fock / fock_eps: loss-term weights, mirroring the
            notebook's ``LAMBDA_V`` / ``LAMBDA_FOCK_REG`` / ``FOCK_REG_EPS``.
        register_repulsion: whether to add ``model.pop_repulsion_loss()``.
        forward_fn: ``(model, x, targets, ctx) -> (loss, loss_ntp, v_reg,
            fock_reg)``. Supply the run's own ``forward_with_vreg`` so a
            replay matches training exactly.
        batch_provider: ``(n) -> LongTensor`` giving a neutral batch, for
            probes that need input but not a specific capture.
    """

    model: nn.Module
    device: str = "cpu"
    store: Optional[BundleStore] = None
    clip_cfg: Optional[GradClipConfig] = None
    grad_accum: Optional[int] = None
    lambda_v: float = 0.0
    lambda_fock: float = 0.0
    fock_eps: float = 1e-6
    register_repulsion: bool = False
    forward_fn: Optional[Callable[..., Any]] = None
    batch_provider: Optional[Callable[[int], torch.Tensor]] = None
    extras: dict = field(default_factory=dict)

    _REASONS = {
        "store": "loading a capture bundle by step tag",
        "clip_cfg": "computing per-group gradient norms",
        "forward_fn": "replaying the captured forward/backward",
        "batch_provider": "obtaining a neutral input batch",
        "grad_accum": "scaling each microbatch's loss contribution",
    }

    def require(self, *names: str) -> None:
        """Assert that each named field was supplied; raise if not.

        Raises:
            MissingContextError: naming the field and what it is needed for,
                rather than failing later with an opaque ``NoneType`` error
                or -- worse -- silently doing nothing.
        """
        missing = [n for n in names if getattr(self, n, None) is None]
        if missing:
            detail = "; ".join(
                f"{n!r} (needed for {self._REASONS.get(n, 'this probe')})"
                for n in missing
            )
            raise MissingContextError(
                f"ProbeContext is missing {detail}. Supply it explicitly -- "
                f"probes never fall back to globals or to a silent no-op."
            )

    def neutral_batch(self, n: int) -> torch.Tensor:
        """A fixed, capture-independent batch, for probes that just need
        *some* input to drive a forward pass."""
        self.require("batch_provider")
        return self.batch_provider(n).to(self.device)
