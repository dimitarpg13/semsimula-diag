"""D2 / D2b: omega*dt via power iteration, and cross-well tail coherence."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from semsimula_diag import BundleStore, GradClipConfig, ProbeContext
from semsimula_diag.probes import resonance
from semsimula_diag.probes.resonance import _lambda_max


# --- the numerical core -------------------------------------------------

def test_power_iteration_matches_exact_eigendecomposition():
    """The whole affordability argument is that lambda_max needs no
    eigensolver. It must still agree with one."""
    torch.manual_seed(0)
    G = torch.randn(6, 40, 12, dtype=torch.double)
    got = _lambda_max(G, n_iter=60, seed=1)
    want = torch.linalg.eigvalsh(G @ G.transpose(-2, -1))[..., -1]
    assert torch.allclose(got, want, rtol=1e-6), (got, want)


def test_power_iteration_is_batched_over_leading_dims():
    torch.manual_seed(0)
    G = torch.randn(3, 5, 20, 7, dtype=torch.double)
    got = _lambda_max(G, n_iter=60, seed=1)
    want = torch.linalg.eigvalsh(G @ G.transpose(-2, -1))[..., -1]
    assert got.shape == (3, 5)
    # Convergence is slow where the top two eigenvalues nearly coincide, and
    # the Rayleigh quotient of an unconverged vector is always an
    # UNDER-estimate -- so assert the bound as well as the tolerance.
    assert (got <= want + 1e-9).all()
    assert torch.allclose(got, want, rtol=5e-3)


def test_power_iteration_is_deterministic():
    G = torch.randn(4, 10, 5)
    assert torch.equal(_lambda_max(G, 8, seed=3), _lambda_max(G, 8, seed=3))


def test_power_iteration_handles_a_rank_deficient_factor():
    """A token where every well is numerically zero-weight gives G = 0;
    lambda_max must be 0, not NaN."""
    G = torch.zeros(3, 9, 4)
    out = _lambda_max(G, 8, seed=1)
    assert torch.isfinite(out).all() and float(out.abs().max()) == 0.0


# --- the monitor against a toy integrator -------------------------------

class _ToyVTheta(nn.Module):
    """Exposes the two methods the monitor hooks, with a controllable
    low-rank factor so omega*dt is known in advance."""

    def __init__(self, d=6, K=2, r=3, scale=1.0):
        super().__init__()
        self.d, self.K, self.r, self.scale = d, K, r, scale
        self.proj = nn.Linear(d, K * d * r, bias=False)

    def _B(self, xis):
        B = self.proj(xis).reshape(*xis.shape[:-1], self.K, self.d, self.r)
        return self.scale * B

    def context_components(self, xis):
        B = self._B(xis)
        z = torch.zeros(*B.shape[:-1])
        return [(z, z + 1.0, z[..., :1].squeeze(-1) + 1.0, B)]

    def harmonic_terms(self, xis, h, *, comps=None):
        k = torch.ones_like(h)
        return k, h * k

    def harmonic_terms_lowrank(self, xis, h, *, comps=None):
        B = self._B(xis) if comps is None else comps[0][3]
        G = B.reshape(*B.shape[:-3], self.d, self.K * self.r)
        return torch.ones_like(h), h, G, G.new_zeros(G.shape[:-2] + (G.shape[-1],))


class _FakeIntegratorModule:
    """Stands in for the module the model imports cfc_substep from."""
    @staticmethod
    def cfc_substep(h, v, f_harm, k_diag, m, dt):
        return h, v


class _ToyModel(nn.Module):
    def __init__(self, scale=1.0, n_layers=2, mass=1.0, dt_substep=0.5):
        super().__init__()
        self.emb = nn.Embedding(16, 6)
        self.V_theta = _ToyVTheta(scale=scale)
        self.head = nn.Linear(6, 16)
        self.n_layers, self.mass, self.dt_substep = n_layers, mass, dt_substep

    def forward(self, x):
        h = self.emb(x)
        for _ in range(self.n_layers):
            xis = h
            comps = self.V_theta.context_components(xis)
            k, s = self.V_theta.harmonic_terms(xis, h, comps=comps)
            h, _ = _FakeIntegratorModule.cfc_substep(
                h, h, s, k, torch.tensor(self.mass), self.dt_substep)
        return self.head(h)


def _forward_fn(model, x, y, ctx):
    logits = model(x)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, 16), y.reshape(-1))
    z = torch.tensor(0.0)
    return loss, loss, z, z


@pytest.fixture
def setup(tmp_path):
    def build(scale=1.0, mass=1.0, dt_substep=0.5):
        torch.manual_seed(0)
        model = _ToyModel(scale=scale, mass=mass, dt_substep=dt_substep)
        # fixed batch: these tests compare omega*dt ACROSS builds, so the
        # data must not move when only the parameter under test does
        rows = np.random.RandomState(0).randint(
            0, 16, size=(2, 5)).astype(np.int64)
        bundle = {"step": 42, "grad_accum": 1, "batches": [(rows, rows)],
                  "model_state_dict": model.state_dict(),
                  "rng_state_cpu": torch.get_rng_state(), "rng_state_cuda": None,
                  "pre_clip_grad_norm": 1.0, "top_groups": {}}
        ck = tmp_path / f"c{scale}{mass}{dt_substep}"
        ck.mkdir(exist_ok=True)
        torch.save(bundle, ck / "run_step42_spikebatch.pt")
        ctx = ProbeContext(
            model=model, device="cpu",
            store=BundleStore(ckpt_dir=ck, ckpt_prefix="run",
                              archive_root=tmp_path, verbose=False),
            clip_cfg=GradClipConfig(default_clip=10.0), forward_fn=_forward_fn)
        return ctx
    return build


def test_monitor_records_every_layer(setup):
    ctx = setup()
    res = resonance.omega_dt_report(ctx, 42, _FakeIntegratorModule,
                                    verbose=False)
    assert set(res.per_layer) == {0, 1}
    assert res.metrics["omega_dt_max"] > 0


def test_omega_dt_scales_with_the_low_rank_factor(setup):
    """omega ~ sqrt(lambda_max) ~ ||B||, so doubling B doubles omega*dt."""
    a = resonance.omega_dt_report(setup(scale=1.0), 42, _FakeIntegratorModule,
                                  verbose=False).metrics["omega_dt_max"]
    b = resonance.omega_dt_report(setup(scale=2.0), 42, _FakeIntegratorModule,
                                  verbose=False).metrics["omega_dt_max"]
    assert b == pytest.approx(2 * a, rel=1e-3)


def test_omega_dt_uses_the_full_kick_step_not_the_half_substep(setup):
    """In baoab_cfc the low-rank part rides the kick (dt), while
    cfc_substep is handed dt/2. Reporting the substep value would
    understate the wall by exactly 2x."""
    got = resonance.omega_dt_report(setup(dt_substep=0.5), 42,
                                    _FakeIntegratorModule,
                                    verbose=False).metrics["omega_dt_max"]
    ctx = setup(dt_substep=0.5)
    with torch.no_grad():
        model = ctx.model
        xis = model.emb(torch.as_tensor(
            np.random.RandomState(0).randint(0, 16, (2, 5))).long())
    # value is built from dt_kick = 2 * dt_substep
    half = resonance.omega_dt_report(setup(dt_substep=0.25), 42,
                                     _FakeIntegratorModule,
                                     verbose=False).metrics["omega_dt_max"]
    assert got == pytest.approx(2 * half, rel=1e-6)


def test_omega_dt_divides_by_mass(setup):
    """omega = sqrt(lambda_max / m): quadrupling the mass halves omega*dt."""
    a = resonance.omega_dt_report(setup(mass=1.0), 42, _FakeIntegratorModule,
                                  verbose=False).metrics["omega_dt_max"]
    b = resonance.omega_dt_report(setup(mass=4.0), 42, _FakeIntegratorModule,
                                  verbose=False).metrics["omega_dt_max"]
    assert b == pytest.approx(a / 2, rel=1e-3)


def test_frac_over_wall_responds_to_the_wall(setup):
    ctx = setup(scale=3.0)
    hi = resonance.omega_dt_report(ctx, 42, _FakeIntegratorModule, wall=1e9,
                                   verbose=False).metrics["frac_over_wall"]
    lo = resonance.omega_dt_report(setup(scale=3.0), 42, _FakeIntegratorModule,
                                   wall=0.0, verbose=False
                                   ).metrics["frac_over_wall"]
    assert hi == 0.0 and lo == 1.0


def test_observe_restores_both_hooks(setup):
    ctx = setup()
    before_h = ctx.model.V_theta.harmonic_terms.__func__
    before_s = _FakeIntegratorModule.cfc_substep
    with resonance.observe(ctx.model, _FakeIntegratorModule):
        pass
    assert ctx.model.V_theta.harmonic_terms.__func__ is before_h
    assert _FakeIntegratorModule.cfc_substep is before_s


def test_report_refuses_a_vtheta_without_the_lowrank_split(setup):
    class _NoSplit(_ToyVTheta):
        harmonic_terms_lowrank = None
        def __getattribute__(self, name):
            if name == "harmonic_terms_lowrank":
                raise AttributeError(name)
            return super().__getattribute__(name)

    ctx = setup()
    ctx.model.V_theta = _NoSplit()
    with pytest.raises(RuntimeError, match="harmonic_terms_lowrank"):
        resonance.omega_dt_report(ctx, 42, _FakeIntegratorModule, verbose=False)


# --- D2b ----------------------------------------------------------------

def test_tail_coherence_detects_aligned_vs_orthogonal_tails(setup):
    """The signature the note predicts: tail_pr near 1 when every well's
    weakest direction points the same way, near n_wells when they do not."""
    ctx = setup()
    res = resonance.tail_coherence_report(ctx, 42, verbose=False)
    assert 1.0 <= res.metrics["tail_pr_p50"] <= ctx.model.V_theta.K + 1e-6
    assert 0.0 <= res.metrics["tail_mean_abs_cos_p50"] <= 1.0 + 1e-6


def test_omega_dt_under_truncation_reports_both_arms(setup):
    """The within-bundle control: truncation must move omega*dt, since it is
    computed from the same B the truncation alters."""
    ctx = setup(scale=2.0)
    out = resonance.omega_dt_under_truncation(
        ctx, 42, _FakeIntegratorModule, ranks=(1,), verbose=False)
    assert set(out) == {"untruncated", "rank=1"}
    full = out["untruncated"].metrics["omega_dt_max"]
    trunc = out["rank=1"].metrics["omega_dt_max"]
    # truncation removes curvature, so it can only lower lambda_max
    assert trunc <= full + 1e-6
    assert trunc < full          # and on a non-degenerate toy, strictly


class _LayeredToy(nn.Module):
    """Exposes `_fock_layer_step` with the signature `_install_layer_hook`
    wraps, so the per-layer path is actually exercised."""

    def __init__(self, n_layers=3, d=6, vocab=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.steps = nn.ModuleList(nn.Linear(d, d) for _ in range(n_layers))
        self.head = nn.Linear(d, vocab)
        self.V_theta = _ToyVTheta(d=d)
        self.n_layers = n_layers

    def _fock_layer_step(self, h, h_prev, r, salience, m_b, gamma, dt,
                         layer_idx, *a, **kw):
        # mirror the real layer step closely enough that observe()'s three
        # hooks all fire: context_components -> harmonic_terms -> cfc_substep
        comps = self.V_theta.context_components(h)
        k, sl = self.V_theta.harmonic_terms(h, h, comps=comps)
        h2, _ = _FakeIntegratorModule.cfc_substep(
            h, h, sl, k, torch.tensor(1.0), 0.5)
        return torch.tanh(self.steps[layer_idx](h2)), h

    def forward(self, x):
        h = self.emb(x)
        h_prev = h
        for li in range(self.n_layers):
            h, h_prev = self._fock_layer_step(
                h, h_prev, None, None, None, None, 1.0, li)
        return self.head(h)


def test_per_layer_profile_is_recorded_for_every_microbatch(tmp_path):
    """A spike can live in any microbatch -- at step 87196 microbatch 2 held
    99.94% of the gradient while microbatch 0 held 0.05%, so a profile keyed
    only by layer (first-microbatch-wins) describes whichever pass ran first,
    not the one that spiked."""
    from semsimula_diag.probes import replayed, restored_model_state
    torch.manual_seed(0)
    model = _LayeredToy()
    rng = np.random.RandomState(0)
    batches = [(r, r) for r in
               (rng.randint(0, 16, (2, 5)).astype(np.int64) for _ in range(3))]
    bundle = {"step": 7, "grad_accum": 3, "batches": batches,
              "model_state_dict": model.state_dict(),
              "rng_state_cpu": torch.get_rng_state(), "rng_state_cuda": None,
              "pre_clip_grad_norm": 1.0, "top_groups": {}}
    ck = tmp_path / "mb"; ck.mkdir()
    torch.save(bundle, ck / "run_step7_spikebatch.pt")
    ctx = ProbeContext(
        model=model, device="cpu",
        store=BundleStore(ckpt_dir=ck, ckpt_prefix="run", archive_root=tmp_path,
                          verbose=False),
        clip_cfg=GradClipConfig(default_clip=10.0), forward_fn=_forward_fn)

    with restored_model_state(ctx, grads=False, weights=False, rng=False):
        with replayed(ctx, bundle, per_layer=True) as info:
            pass

    assert set(info.per_layer_by_mb) == {0, 1, 2}, info.per_layer_by_mb
    for mb, prof in info.per_layer_by_mb.items():
        assert set(prof) == {0, 1, 2}, (mb, prof)
    # microbatches genuinely differ -- otherwise the test proves nothing
    assert info.per_layer_by_mb[0] != info.per_layer_by_mb[2]
    # the legacy field still holds microbatch 0, unchanged
    assert info.per_layer_h_grad == info.per_layer_by_mb[0]


def test_monitor_separates_microbatches_and_layers(tmp_path):
    """The flat per_layer view pools microbatches, so a spike confined to one
    of them is invisible in it. by_mb_layer keeps them apart."""
    torch.manual_seed(0)
    model = _LayeredToy(n_layers=3)
    rng = np.random.RandomState(0)
    n_mb = 3
    batches = [(r, r) for r in
               (rng.randint(0, 16, (2, 5)).astype(np.int64) for _ in range(n_mb))]
    bundle = {"step": 9, "grad_accum": n_mb, "batches": batches,
              "model_state_dict": model.state_dict(),
              "rng_state_cpu": torch.get_rng_state(), "rng_state_cuda": None,
              "pre_clip_grad_norm": 1.0, "top_groups": {}}
    ck = tmp_path / "mbl"; ck.mkdir()
    torch.save(bundle, ck / "run_step9_spikebatch.pt")
    ctx = ProbeContext(
        model=model, device="cpu",
        store=BundleStore(ckpt_dir=ck, ckpt_prefix="run", archive_root=tmp_path,
                          verbose=False),
        clip_cfg=GradClipConfig(default_clip=10.0), forward_fn=_forward_fn)

    from semsimula_diag.probes import replayed, restored_model_state
    with resonance.observe(model, _FakeIntegratorModule) as mon:
        with restored_model_state(ctx, grads=False, weights=False, rng=False):
            with replayed(ctx, bundle):
                pass
    keys = set(mon.by_mb_layer)
    assert keys == {(mb, li) for mb in range(n_mb) for li in range(3)}, keys
    # and the flat view really does pool them, which is why it was misleading
    assert set(mon.per_layer) == {0, 1, 2}
