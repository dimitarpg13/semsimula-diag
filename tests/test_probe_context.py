"""ProbeContext must fail loudly on missing configuration.

This is the direct lesson from the CUDA debugging session: the notebook's
`_cts_group_params` reads its settings via `globals().get(...)`, which
returns None rather than raising, so an unset CLIP_THEN_SUM_GROUPS silently
degraded clip_then_sum into sum_then_clip and reintroduced the exact bug
Mitigations SS45.3-45.4 exists to fix. It cost a full round-trip on real
hardware before the symptom was recognised.
"""
from __future__ import annotations

import pytest
import torch.nn as nn

from semsimula_diag.probes import MissingContextError, ProbeContext


def _ctx(**kw):
    return ProbeContext(model=nn.Linear(2, 2), **kw)


def test_require_passes_when_supplied():
    ctx = _ctx(grad_accum=4)
    ctx.require("model", "grad_accum")     # must not raise


def test_require_raises_naming_the_field_and_its_purpose():
    with pytest.raises(MissingContextError) as e:
        _ctx().require("store")
    msg = str(e.value)
    assert "'store'" in msg
    assert "capture bundle" in msg, msg


def test_require_reports_every_missing_field_at_once():
    """One round-trip per missing global is exactly the failure mode that
    made the Colab debugging slow; report them together."""
    with pytest.raises(MissingContextError) as e:
        _ctx().require("store", "clip_cfg", "forward_fn")
    msg = str(e.value)
    for field in ("'store'", "'clip_cfg'", "'forward_fn'"):
        assert field in msg, msg


def test_require_never_silently_noops():
    ctx = _ctx()
    assert ctx.store is None
    with pytest.raises(MissingContextError):
        ctx.require("store")


def test_neutral_batch_requires_a_provider():
    with pytest.raises(MissingContextError, match="batch_provider"):
        _ctx().neutral_batch(2)


def test_neutral_batch_uses_the_provider_and_moves_to_device():
    import torch
    ctx = _ctx(batch_provider=lambda n: torch.ones(n, 3, dtype=torch.long))
    x = ctx.neutral_batch(2)
    assert x.shape == (2, 3)
    assert x.device.type == "cpu"
