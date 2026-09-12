"""Stiffness / curvature probes, exercised against a stub V_theta.

The real anisotropic Gaussian lives in semsimula-paper and needs a 77M-param
model to drive, so these tests stub the two hook points the probes actually
depend on -- `context_components` and `harmonic_terms` -- with tensors whose
participation ratio and quantiles are known in closed form. That tests THIS
package's logic (recording, restoration, PR, quantiles) rather than the
model's.

Validated separately against the real step-87196 bundle; see MIGRATION.md.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from semsimula_diag import ProbeContext
from semsimula_diag.probes import stiffness


class _StubVTheta(nn.Module):
    """Exposes the hook points with B factors of a chosen spectrum."""

    def __init__(self, singular_values, n_wells=3, d=6):
        super().__init__()
        self.singular_values = list(singular_values)
        self.n_wells, self.d = n_wells, d
        self.calls = 0

    def _B(self):
        r = len(self.singular_values)
        # Build B = U diag(s) with orthonormal U so svdvals(B) == s exactly.
        U = torch.linalg.qr(torch.randn(self.d, r))[0]        # (d, r)
        s = torch.tensor(self.singular_values, dtype=torch.float32)
        return (U * s).expand(self.n_wells, self.d, r).clone()

    def context_components(self, xis):
        self.calls += 1
        return [(None, None, None, self._B())]

    def harmonic_terms(self, xis, h):
        self.calls += 1
        return torch.full((4, 5), 2.0), None


class _StubModel(nn.Module):
    def __init__(self, vtheta):
        super().__init__()
        self.V_theta = vtheta
        self.cfg = type("cfg", (), {"dt": 1.0, "integrator": "verlet",
                                    "vtheta_analytic_force": False})()
        self.forward_calls = 0

    def forward(self, x):
        self.V_theta.context_components(None)
        self.V_theta.harmonic_terms(None, None)
        self.forward_calls += 1
        return x

    def compute_mass(self, x):
        return torch.full((2,), 0.5)


def _ctx(sv=(2.0, 1.0, 1.0, 1.0)):
    return ProbeContext(model=_StubModel(_StubVTheta(sv)), device="cpu")


def test_spectrum_participation_ratio_matches_closed_form():
    """PR = (sum s^2)^2 / sum s^4, in [1, rank]."""
    sv = (2.0, 1.0, 1.0, 1.0)
    ctx = _ctx(sv)
    res = stiffness.sigma_lr_spectrum_report(ctx, torch.zeros(1))
    s2 = [v ** 2 for v in sv]
    expected = sum(s2) ** 2 / sum(v ** 2 for v in s2)
    assert res.metrics["rank"] == 4
    assert res.metrics["pr_p50"] == pytest.approx(expected, rel=1e-4)
    assert 1.0 <= res.metrics["pr_p50"] <= 4.0


def test_degenerate_spectrum_gives_participation_ratio_one():
    """A single live direction is PR == 1 -- full spectral collapse."""
    res = stiffness.sigma_lr_spectrum_report(_ctx((1.0, 0.0, 0.0, 0.0)),
                                             torch.zeros(1))
    assert res.metrics["pr_p50"] == pytest.approx(1.0, rel=1e-4)


def test_flat_spectrum_gives_participation_ratio_equal_to_rank():
    res = stiffness.sigma_lr_spectrum_report(_ctx((1.0, 1.0, 1.0, 1.0)),
                                             torch.zeros(1))
    assert res.metrics["pr_p50"] == pytest.approx(4.0, rel=1e-4)
    assert res.raw["spectrum"] == pytest.approx([1.0, 1.0, 1.0, 1.0], rel=1e-4)


def test_frobenius_norm_is_reported_so_cap_binding_can_be_checked():
    """fro_p50 is what says whether rank is a redistribution knob at all."""
    sv = (3.0, 4.0)          # ||B||_F = 5
    res = stiffness.sigma_lr_spectrum_report(_ctx(sv), torch.zeros(1))
    assert res.metrics["fro_p50"] == pytest.approx(5.0, rel=1e-4)


def test_sigma_lr_report_returns_sigma_max_squared():
    res = stiffness.sigma_lr_report(_ctx((3.0, 1.0)), torch.zeros(1))
    assert res.metrics["median"] == pytest.approx(9.0, rel=1e-4)
    assert res.metrics["max"] == pytest.approx(9.0, rel=1e-4)


def test_stiffness_report_flags_the_instability_wall():
    """omega*dt = sqrt(k/m)*dt; k=2, m=0.5, dt=1 -> exactly 2.0 == unstable."""
    ctx = _ctx()
    res = stiffness.stiffness_report(ctx, torch.zeros(1))
    assert res.metrics["median"] == pytest.approx(2.0, rel=1e-4)
    assert res.metrics["frac_marginal"] == pytest.approx(1.0)


def test_hooks_are_restored_even_when_the_forward_raises():
    """The non-pollution invariant: a probe must never leave the model
    monkeypatched, or every later step silently records into a dead list."""
    ctx = _ctx()
    original = ctx.model.V_theta.context_components

    def _boom(x):
        raise RuntimeError("forward blew up")

    ctx.model.forward = _boom
    with pytest.raises(RuntimeError, match="forward blew up"):
        stiffness.sigma_lr_report(ctx, torch.zeros(1))
    assert ctx.model.V_theta.context_components == original


def test_training_mode_is_restored():
    ctx = _ctx()
    ctx.model.train()
    stiffness.sigma_lr_report(ctx, torch.zeros(1))
    assert ctx.model.training, "probe left the model in eval mode"


def test_stiffness_report_restores_integrator_config():
    """It forces baoab_cfc to measure the explicit-kick wall; that must not
    persist into the caller's next real step."""
    ctx = _ctx()
    before = (ctx.model.cfg.integrator, ctx.model.cfg.vtheta_analytic_force)
    stiffness.stiffness_report(ctx, torch.zeros(1))
    assert (ctx.model.cfg.integrator,
            ctx.model.cfg.vtheta_analytic_force) == before


def test_empty_low_rank_factor_is_skipped_not_crashed():
    class _EmptyB(_StubVTheta):
        def context_components(self, xis):
            return [(None, None, None, torch.zeros(2, 6, 0))]

    ctx = ProbeContext(model=_StubModel(_EmptyB((1.0,))), device="cpu")
    res = stiffness.sigma_lr_report(ctx, torch.zeros(1))
    assert res.metrics["n_samples"] == 0


def test_checkpoint_sweeps_require_their_context():
    from semsimula_diag.probes import MissingContextError
    with pytest.raises(MissingContextError, match="store"):
        stiffness.spectrum_across_checkpoints(_ctx())
    with pytest.raises(MissingContextError, match="store"):
        stiffness.bracket_precision_lr_max(_ctx())
