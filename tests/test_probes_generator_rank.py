"""generator_rank against a V_theta with planted structure.

The probe answers "how many of the xi input directions does the well-bank
generator actually need". These tests plant a known answer and check it is
recovered, and cover the zero-matrix case that a freshly built model
genuinely produces (a_proj and w_proj are zero-initialised).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from semsimula_diag import ProbeContext
from semsimula_diag.probes import generator_rank


class _Bank(nn.Module):
    def __init__(self, xi_dim: int, out_mu: int, out_B: int):
        super().__init__()
        self.mu_proj = nn.Linear(xi_dim, out_mu, bias=False)
        self.a_proj = nn.Linear(xi_dim, out_mu, bias=False)
        self.B_proj = nn.Linear(xi_dim, out_B, bias=False)


class _Model(nn.Module):
    def __init__(self, xi_dim=96, out_mu=128, out_B=256):
        super().__init__()
        self.V_theta = _Bank(xi_dim, out_mu, out_B)


def _plant(linear: nn.Linear, rank: int, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    o, i = linear.weight.shape
    u = torch.randn(o, rank, generator=g)
    v = torch.randn(rank, i, generator=g)
    with torch.no_grad():
        linear.weight.copy_((u @ v) / rank ** 0.5)


@pytest.mark.parametrize("planted", [5, 12, 31])
def test_recovers_a_planted_rank(planted):
    """r@0.99 is an ENERGY rank, so it never exceeds the true rank and may
    sit slightly below it -- a random rank-31 matrix carries 99% of its
    energy in about 29 directions. The probe must bound the true rank from
    below without ever overstating it, and must be far below the ambient
    input dimension (96 here) so it is genuinely detecting structure."""
    m = _Model()
    _plant(m.V_theta.B_proj, rank=planted)
    res = generator_rank.generator_spectrum(
        ProbeContext(model=m, device="cpu"), fractions=(0.99, 0.999),
        verbose=False)
    r99 = res.metrics["B_proj.weight.r@0.99"]
    r999 = res.metrics["B_proj.weight.r@0.999"]
    assert r99 <= planted, "energy rank must never exceed the true rank"
    assert r99 >= 0.7 * planted, "should recover most of the planted rank"
    assert r999 <= planted
    assert r99 < 96, "must be well below the ambient input dimension"


def test_zero_matrix_reports_rank_zero_not_nan():
    """a_proj and w_proj are zero-initialised in the real bank."""
    m = _Model()
    with torch.no_grad():
        m.V_theta.a_proj.weight.zero_()
    res = generator_rank.generator_spectrum(
        ProbeContext(model=m, device="cpu"), fractions=(0.9, 0.99),
        verbose=False)
    assert res.metrics["a_proj.weight.r@0.9"] == 0
    assert res.metrics["a_proj.weight.pr"] == 0.0
    for v in res.metrics.values():
        assert not torch.isnan(torch.tensor(float(v)))


def test_full_rank_truncation_is_a_no_op():
    m = _Model()
    _plant(m.V_theta.B_proj, rank=40)
    before = {n: p.detach().clone() for n, p in m.V_theta.named_parameters()}
    calls = []

    def fake_eval():
        calls.append(1)
        return 2.0 + 0.001 * len(calls)

    res = generator_rank.truncation_sweep(
        ProbeContext(model=m, device="cpu"), fake_eval,
        ranks=(1024,), verbose=False)
    # weights restored regardless
    for n, p in m.V_theta.named_parameters():
        assert torch.allclose(p, before[n])
    assert res.raw["rows"][0]["rank"] == 1024


def test_weights_are_restored_even_when_evaluate_raises():
    m = _Model()
    _plant(m.V_theta.B_proj, rank=8)
    before = {n: p.detach().clone() for n, p in m.V_theta.named_parameters()}

    def boom():
        raise RuntimeError("eval blew up")

    with pytest.raises(RuntimeError):
        generator_rank.truncation_sweep(
            ProbeContext(model=m, device="cpu"), boom, ranks=(4,),
            verbose=False)
    for n, p in m.V_theta.named_parameters():
        assert torch.allclose(p, before[n]), f"{n} left truncated"


def test_missing_vtheta_is_an_explicit_error():
    with pytest.raises(ValueError, match="no V_theta"):
        generator_rank.generator_spectrum(
            ProbeContext(model=nn.Linear(4, 4), device="cpu"), verbose=False)
