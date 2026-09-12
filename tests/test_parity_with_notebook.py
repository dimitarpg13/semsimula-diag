"""Parity gate: this package vs the original semsimula-paper implementation.

Diagnostic Programme SS11.5 asks that each extraction be *provably*
behaviour-preserving rather than "should be equivalent". These tests run the
ported code and the original side by side on identical inputs and assert the
outputs are bit-equal.

Skipped automatically when the semsimula-paper checkout isn't alongside this
repo, so CI without it still passes; set SEMSIMULA_PAPER to point elsewhere.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_DEFAULT = Path(__file__).resolve().parents[2] / 'semsimula-paper'
_PAPER = Path(os.environ.get('SEMSIMULA_PAPER', _DEFAULT))
_SCALEUP = _PAPER / 'notebooks' / 'conservative_arch' / 'scaleup'

pytestmark = pytest.mark.skipif(
    not (_SCALEUP / 'grad_clip_utils.py').exists(),
    reason=f'semsimula-paper not found at {_PAPER} (set SEMSIMULA_PAPER)',
)

if (_SCALEUP / 'grad_clip_utils.py').exists():
    sys.path.insert(0, str(_SCALEUP))
    import grad_clip_utils as orig  # noqa: E402
else:                                # pragma: no cover
    orig = None

from semsimula_diag import clipping as new  # noqa: E402

# the live run's real override table; order-sensitive per Mitigations SS49.8
OVERRIDES = {'log_tau': 0.3, 'creation_gate': 0.3,
             'reverse_channel_scale': 0.1, 'depth_code': 0.25}
EXCL = frozenset({'override:reverse_channel_scale'})
NAMES = ['E', 'P', 'score_head', 'V_theta.depth_code',
         'creation_gate_qkv.log_tau', 'creation_gate_qkv.W_Q',
         'reverse_channel_scale', 'unmatched.thing']


def _build():
    m = nn.Module()
    m.E = nn.Parameter(torch.randn(8))
    m.P = nn.Parameter(torch.randn(8))
    m.score_head = nn.Parameter(torch.randn(4))
    m.V_theta = nn.Module()
    m.V_theta.depth_code = nn.Parameter(torch.randn(6))
    m.creation_gate_qkv = nn.Module()
    m.creation_gate_qkv.log_tau = nn.Parameter(torch.randn(3))
    m.creation_gate_qkv.W_Q = nn.Parameter(torch.randn(5))
    m.reverse_channel_scale = nn.Parameter(torch.randn(2))
    return m


def _cfgs(**kw):
    o = orig.GradClipConfig(default_clip=1.0, overrides=OVERRIDES,
                            watchdog_exclude_groups=EXCL)
    n = new.GradClipConfig(default_clip=1.0, overrides=OVERRIDES,
                           watchdog_exclude_groups=EXCL, **kw)
    return o, n


def _pair():
    torch.manual_seed(0)
    a = _build()
    b = _build()
    b.load_state_dict(a.state_dict())
    g = {n: torch.randn_like(p) for n, p in a.named_parameters()}
    for m in (a, b):
        for n, p in m.named_parameters():
            p.grad = g[n].clone()
    return a, b


def test_assign_clip_group_parity():
    o_cfg, n_cfg = _cfgs()
    for nm in NAMES:
        assert orig.assign_clip_group(nm, o_cfg) == new.assign_clip_group(nm, n_cfg), nm
    # the SS49.8 ordering trap specifically
    assert new.assign_clip_group('creation_gate_qkv.log_tau', n_cfg)[0] == 'override:log_tau'


def test_per_group_grad_norms_parity():
    o_cfg, n_cfg = _cfgs()
    a, b = _pair()
    x, y = orig.per_group_grad_norms(a, o_cfg), new.per_group_grad_norms(b, n_cfg)
    assert set(x) == set(y)
    for k in x:
        assert abs(x[k] - y[k]) < 1e-12, (k, x[k], y[k])


def test_clip_grads_per_group_parity_including_mutated_grads():
    o_cfg, n_cfg = _cfgs()
    a, b = _pair()
    agg_a, pg_a = orig.clip_grads_per_group(a, o_cfg)
    agg_b, pg_b = new.clip_grads_per_group(b, n_cfg)
    assert abs(float(agg_a) - float(agg_b)) < 1e-12
    assert set(pg_a) == set(pg_b)
    for k in pg_a:
        assert abs(pg_a[k] - pg_b[k]) < 1e-12, (k, pg_a[k], pg_b[k])
    for (n, p1), (_, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p1.grad, p2.grad), f'post-clip grad differs for {n}'


def test_clip_then_sum_parity_with_notebook_cts_trio():
    """ClipThenSum vs Cell 6d's _cts_* trio, replicated verbatim here."""
    CTS, THR, ACCUM = frozenset({'E', 'P'}), 0.5, 4
    o_cfg, n_cfg = _cfgs(clip_then_sum_groups=CTS, clip_then_sum_threshold=THR)

    def nb_group_params(mdl):
        out = {}
        for n, p in mdl.named_parameters():
            if not p.requires_grad:
                continue
            key, _ = orig.assign_clip_group(n, o_cfg)
            if key in CTS:
                out.setdefault(key, []).append(p)
        return out

    def nb_apply(params, running):
        for gp in params.values():
            if not any(p.grad is not None for p in gp):
                continue
            nn.utils.clip_grad_norm_(gp, THR)
            for p in gp:
                if p.grad is None:
                    continue
                pid = id(p)
                c = p.grad.detach().clone()
                running[pid] = running[pid] + c if pid in running else c
                p.grad.zero_()

    def nb_splice(params, running):
        for gp in params.values():
            for p in gp:
                if id(p) in running:
                    p.grad = running[id(p)]

    torch.manual_seed(1)
    a, b = _build(), _build()
    b.load_state_dict(a.state_dict())
    micro = [{n: torch.randn_like(p) for n, p in a.named_parameters()}
             for _ in range(ACCUM)]

    nb_params, nb_running = nb_group_params(a), {}
    cts = new.ClipThenSum(b, n_cfg)
    assert set(nb_params) == set(cts.params)

    for mb in micro:
        for m in (a, b):
            for n, p in m.named_parameters():
                p.grad = mb[n].clone()
        nb_apply(nb_params, nb_running)
        cts.apply_microbatch()
    nb_splice(nb_params, nb_running)
    cts.splice_back()

    for (n, p1), (_, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p1.grad, p2.grad), f'clip_then_sum grad differs for {n}'
