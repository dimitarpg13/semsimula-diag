"""Correctness tests for semsimula_diag.clipping.

The first four are ported unchanged in substance from
``test_grad_clip_utils.py`` in the semsimula-paper repo (the module this one
renames in place, Diagnostic Programme SS11.6 step 1), so the port is held to
the same behaviour-preserving standard the note asks for.

The clip_then_sum tests are new: that code had no test at all while it lived
inline in Cell 6, and its whole reason for existing (Mitigations SS45.3-45.4)
is that clip-order changes the answer -- so the key test asserts exactly that,
rather than merely that it runs.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from semsimula_diag.clipping import (
    ClipThenSum,
    GradClipConfig,
    assign_clip_group,
    clip_grads_per_group,
    clip_then_sum_params,
    per_group_grad_norms,
)


class _ToyModel(nn.Module):
    """Names chosen to exercise: an override match ('depth_code' inside
    'V_theta.depth_code'), a plain top-level group ('E'), and a group with no
    override match at all ('score_head')."""

    def __init__(self):
        super().__init__()
        self.E = nn.Parameter(torch.zeros(4))
        self.score_head = nn.Parameter(torch.zeros(4))
        # named_parameters() only descends into registered submodules --
        # V_theta must actually be an nn.Module for 'V_theta.depth_code' to
        # show up with that dotted name.
        self.V_theta = nn.Module()
        self.V_theta.depth_code = nn.Parameter(torch.zeros(4))


def _cfg(**kw):
    base = dict(
        default_clip=1.0,
        overrides={'depth_code': 0.25, 'reverse_channel_scale': 0.1},
        watchdog_exclude_groups=frozenset({'override:reverse_channel_scale'}),
    )
    base.update(kw)
    return GradClipConfig(**base)


# --- ported from test_grad_clip_utils.py ------------------------------------

def test_assign_clip_group_override_match():
    key, thr = assign_clip_group('V_theta.depth_code', _cfg())
    assert key == 'override:depth_code'
    assert thr == 0.25


def test_assign_clip_group_default_fallback():
    cfg = _cfg()
    assert assign_clip_group('score_head', cfg) == ('score_head', 1.0)
    assert assign_clip_group('E.weight', cfg) == ('E', 1.0)


def test_per_group_grad_norms_matches_manual_computation():
    cfg, model = _cfg(), _ToyModel()
    with torch.no_grad():
        model.E.grad = torch.full_like(model.E, 3.0)                    # norm 6.0
        model.score_head.grad = torch.full_like(model.score_head, 1.0)  # norm 2.0
        model.V_theta.depth_code.grad = torch.full_like(
            model.V_theta.depth_code, 5.0)                              # norm 10.0

    out = per_group_grad_norms(model, cfg)
    assert set(out) == {'E', 'score_head', 'override:depth_code'}
    assert out['E'] == pytest.approx(6.0, abs=1e-5)
    assert out['score_head'] == pytest.approx(2.0, abs=1e-5)
    assert out['override:depth_code'] == pytest.approx(10.0, abs=1e-5)
    # read-only: grads must be untouched
    assert torch.allclose(model.E.grad, torch.full_like(model.E, 3.0))


def test_clip_grads_per_group_clips_and_excludes_from_aggregate():
    cfg = GradClipConfig(
        default_clip=1.0,
        overrides={'reverse_channel_scale': 0.1},
        watchdog_exclude_groups=frozenset({'override:reverse_channel_scale'}),
    )
    model = nn.Module()
    model.E = nn.Parameter(torch.zeros(4))
    model.reverse_channel_scale = nn.Parameter(torch.zeros(1))
    with torch.no_grad():
        model.E.grad = torch.full_like(model.E, 3.0)          # norm 6.0, clip 1.0
        model.reverse_channel_scale.grad = torch.full_like(
            model.reverse_channel_scale, 100.0)               # clip 0.1, excluded

    agg, per_group = clip_grads_per_group(model, cfg)

    assert float(model.E.grad.norm()) == pytest.approx(1.0, abs=1e-4)
    assert float(model.reverse_channel_scale.grad.norm()) == pytest.approx(0.1, abs=1e-4)
    assert per_group['E'] == pytest.approx(6.0, abs=1e-4)
    assert per_group['override:reverse_channel_scale'] == pytest.approx(100.0, abs=1e-2)
    # aggregate excludes the override group -> E's pre-clip norm alone,
    # NOT sqrt(6^2 + 100^2). This is the documented watchdog blind spot.
    assert float(agg) == pytest.approx(6.0, abs=1e-4)


# --- new: clip_then_sum (Mitigations SS45.3-45.4) ---------------------------

def test_clip_then_sum_inactive_by_default_is_a_true_noop():
    """Pre-SS45.4 configs must fall through to plain accumulation untouched."""
    model, cfg = _ToyModel(), _cfg()
    assert clip_then_sum_params(model, cfg) == {}

    cts = ClipThenSum(model, cfg)
    assert not cts.active
    with torch.no_grad():
        model.E.grad = torch.full_like(model.E, 3.0)
    cts.apply_microbatch()
    cts.splice_back()
    # untouched: no clipping, no zeroing, no splice
    assert torch.allclose(model.E.grad, torch.full_like(model.E, 3.0))


def test_clip_then_sum_threshold_required_when_groups_set():
    with pytest.raises(ValueError, match="clip_then_sum_threshold is required"):
        GradClipConfig(default_clip=1.0, clip_then_sum_groups=frozenset({'E'}))


def test_clip_then_sum_differs_from_sum_then_clip():
    """The whole point of SS45.4: clip order changes the resulting gradient.

    Two microbatches each contributing norm 6.0 in the SAME direction, with a
    per-microbatch threshold of 1.0.
      clip_then_sum: clip each to 1.0, then sum  -> norm 2.0
      sum_then_clip: sum to 12.0, then clip to 1.0 -> norm 1.0
    """
    cfg = GradClipConfig(
        default_clip=1.0,
        clip_then_sum_groups=frozenset({'E'}),
        clip_then_sum_threshold=1.0,
    )
    model = nn.Module()
    model.E = nn.Parameter(torch.zeros(4))

    cts = ClipThenSum(model, cfg)
    assert cts.active
    for _ in range(2):
        with torch.no_grad():
            model.E.grad = torch.full_like(model.E, 3.0)   # norm 6.0
        cts.apply_microbatch()
        # zeroed after folding into the running total, so the caller's own
        # accumulation cannot double-count it
        assert float(model.E.grad.norm()) == pytest.approx(0.0, abs=1e-6)
    cts.splice_back()

    assert float(model.E.grad.norm()) == pytest.approx(2.0, abs=1e-4)

    # contrast: sum_then_clip on the same inputs lands at 1.0
    model2 = nn.Module()
    model2.E = nn.Parameter(torch.zeros(4))
    with torch.no_grad():
        model2.E.grad = torch.full_like(model2.E, 6.0)     # 3.0 + 3.0 summed
    nn.utils.clip_grad_norm_([model2.E], 1.0)
    assert float(model2.E.grad.norm()) == pytest.approx(1.0, abs=1e-4)


def test_clip_then_sum_only_touches_its_own_groups():
    cfg = GradClipConfig(
        default_clip=1.0,
        overrides={'depth_code': 0.25},
        clip_then_sum_groups=frozenset({'E'}),
        clip_then_sum_threshold=1.0,
    )
    model = _ToyModel()
    assert set(clip_then_sum_params(model, cfg)) == {'E'}

    cts = ClipThenSum(model, cfg)
    with torch.no_grad():
        model.E.grad = torch.full_like(model.E, 3.0)
        model.V_theta.depth_code.grad = torch.full_like(model.V_theta.depth_code, 5.0)
    cts.apply_microbatch()
    # depth_code is not a clip_then_sum group -> untouched by the accumulator
    assert float(model.V_theta.depth_code.grad.norm()) == pytest.approx(10.0, abs=1e-4)
