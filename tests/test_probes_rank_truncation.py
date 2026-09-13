"""replay_rank_truncation_ablation against a toy model with a real
low-rank precision structure.

Verified separately: the SVD-truncation math itself (rank>=r_full is a
bit-exact no-op; rank=1 zeros every singular value but the first) is
checked directly against torch.linalg.svdvals in a standalone script, not
duplicated here. These tests exercise the probe's own logic: hook
installation/restoration, the full-vs-truncated comparison, and the
relative_force_error metric's actual sensitivity to truncation.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from semsimula_diag import BundleStore, GradClipConfig, ProbeContext
from semsimula_diag.probes import precision_cap
from semsimula_diag.probes.precision_cap import _svd_truncate


class _ToyVTheta(nn.Module):
    """A well bank whose force depends on B through the SAME einsum shape
    the real AnisotropicMixtureGaussianVTheta uses, so a rank-1 collapse of
    B has a real, checkable effect on the computed force/gradient.

    xi (context) and h (hidden state) live in the SAME dimension `d` here,
    matching the real model where both are read out of the same embedding
    space -- xi_dim is only used to size mu_proj/B_proj's INPUT, not mu's
    own shape, which must match h's dimension for `h - mu` to broadcast.
    """

    def __init__(self, xi_dim=4, d=6, K=2, r=3):
        super().__init__()
        self.d, self.K, self.r = d, K, r
        self.mu_proj = nn.Linear(xi_dim, K * d, bias=False)
        self.B_proj = nn.Linear(xi_dim, K * d * r, bias=False)
        nn.init.normal_(self.B_proj.weight, std=1.0)   # a real spread, not degenerate

    def context_components(self, xi):
        lead = xi.shape[:-1]
        mu = self.mu_proj(xi).view(*lead, self.K, self.d)
        a = torch.ones(*lead, self.K, self.d)
        w = torch.ones(*lead, self.K)
        B = self.B_proj(xi).view(*lead, self.K, self.d, self.r)
        return [(mu, a, w, B)]   # ONE bank, matching the real list-of-tuples shape

    def forward(self, xi, h, comps=None):
        comps = self.context_components(xi) if comps is None else comps
        (mu, a, w, B) = comps[0]
        diff = h.unsqueeze(-2) - mu
        diag_term = (a * diff * diff).sum(-1)
        lr_term = torch.einsum('...kd,...kdr->...kr', diff, B)
        lr_term = (lr_term * lr_term).sum(-1)
        exponent = -0.5 * (diag_term + lr_term)
        bumps = w * torch.exp(exponent)
        return -bumps.sum(-1, keepdim=True)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.V_theta = _ToyVTheta(xi_dim=4, d=6)
        self.E_ctx = nn.Embedding(16, 4)    # produces xi (context), dim 4
        self.E_h = nn.Embedding(16, 6)      # produces h (hidden state), dim 6 == V_theta.d

    def forward(self, x):
        xi = self.E_ctx(x).mean(dim=1)      # (B, 4)
        h = self.E_h(x).mean(dim=1)         # (B, 6)
        v = self.V_theta(xi, h)
        return v.sum()


def _forward_fn(model, x, y, ctx):
    loss = model(x)
    z = torch.tensor(0.0)
    return loss, loss.detach(), z, z


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(0)
    model = _ToyModel()
    rows = np.random.randint(0, 16, size=(2, 5)).astype(np.int64)
    bundle = {
        "step": 42, "grad_accum": 1,
        "batches": [(rows, rows)],
        "model_state_dict": model.state_dict(),
        "rng_state_cpu": torch.get_rng_state(),
        "rng_state_cuda": None,
        "pre_clip_grad_norm": 1.0,
        "top_groups": {},
    }
    ckpt = tmp_path / "c"; ckpt.mkdir()
    torch.save(bundle, ckpt / "run_step42_spikebatch.pt")
    store = BundleStore(ckpt_dir=ckpt, ckpt_prefix="run", archive_root=tmp_path,
                        verbose=False)
    cfg = GradClipConfig(default_clip=10.0)
    ctx = ProbeContext(model=model, device="cpu", store=store, clip_cfg=cfg,
                       forward_fn=_forward_fn)
    return ctx, bundle


def test_full_untruncated_arm_has_zero_relative_error(setup):
    ctx, bundle = setup
    out = precision_cap.replay_rank_truncation_ablation(
        ctx, 42, ranks=(3,), verbose=False)   # 3 == the toy's own full rank
    assert out["full (untruncated)"].metrics["relative_force_error"] == 0.0


def test_rank_equal_to_full_rank_is_also_near_zero_error(setup):
    """rank=r_full truncation is a bit-exact no-op (see _svd_truncate), so
    it should reproduce the untruncated arm almost exactly."""
    ctx, bundle = setup
    out = precision_cap.replay_rank_truncation_ablation(
        ctx, 42, ranks=(3,), verbose=False)
    assert out["rank=3"].metrics["relative_force_error"] < 1e-5


def test_truncation_to_rank_one_moves_the_gradient(setup):
    """The toy's B is initialised with a real (non-degenerate) spread, so
    collapsing it to rank 1 must change the force by a non-trivial amount --
    this is the actual sensitivity the probe exists to detect."""
    ctx, bundle = setup
    out = precision_cap.replay_rank_truncation_ablation(
        ctx, 42, ranks=(1,), verbose=False)
    assert out["rank=1"].metrics["relative_force_error"] > 1e-3


def test_error_grows_monotonically_as_rank_shrinks(setup):
    """Not a mathematical law in general, but true here because the toy's
    B has a genuine spread across all 3 directions with no repeated
    singular values -- a useful sanity check on the metric's direction."""
    ctx, bundle = setup
    out = precision_cap.replay_rank_truncation_ablation(
        ctx, 42, ranks=(1, 2, 3), verbose=False)
    e1 = out["rank=1"].metrics["relative_force_error"]
    e2 = out["rank=2"].metrics["relative_force_error"]
    e3 = out["rank=3"].metrics["relative_force_error"]
    assert e1 >= e2 >= e3 == pytest.approx(0.0, abs=1e-5)


def test_hook_is_removed_after_the_call(setup):
    """Bound methods aren't identity-stable across separate attribute
    accesses (Python creates a fresh wrapper object each access), so
    compare the underlying function via __func__ instead -- that's stable
    and is the right invariant: after restoration, calling
    context_components must run the ORIGINAL class method, not the
    truncation wrapper's closure. (patched_attrs's restore does leave
    context_components as a genuine instance attribute rather than
    deleting it back to a pure class-level lookup -- harmless, since the
    VALUE is functionally identical either way, but worth knowing if you
    go looking for it in __dict__.)"""
    ctx, bundle = setup
    original_func = ctx.model.V_theta.context_components.__func__
    precision_cap.replay_rank_truncation_ablation(ctx, 42, ranks=(1, 2),
                                                  verbose=False)
    assert ctx.model.V_theta.context_components.__func__ is original_func


def test_weights_and_training_mode_restored(setup):
    ctx, bundle = setup
    before = ctx.model.V_theta.B_proj.weight.clone()
    ctx.model.eval()
    precision_cap.replay_rank_truncation_ablation(ctx, 42, ranks=(1,),
                                                  verbose=False)
    assert torch.equal(ctx.model.V_theta.B_proj.weight, before)
    assert not ctx.model.training, "eval mode should have been restored"


def test_zero_width_lowrank_factor_is_left_alone(setup):
    """B.shape[-1] == 0 (rank disabled entirely) must not crash the hook."""
    ctx, bundle = setup
    orig_components = ctx.model.V_theta.context_components

    def _zero_rank_components(xi):
        comps = orig_components(xi)
        mu, a, w, B = comps[0]
        return [(mu, a, w, B[..., :0])]
    ctx.model.V_theta.context_components = _zero_rank_components
    try:
        out = precision_cap.replay_rank_truncation_ablation(
            ctx, 42, ranks=(1,), verbose=False)
        assert "rank=1" in out
    finally:
        ctx.model.V_theta.context_components = orig_components


def test_returns_one_result_per_requested_rank_plus_reference(setup):
    ctx, bundle = setup
    out = precision_cap.replay_rank_truncation_ablation(
        ctx, 42, ranks=(1, 2, 3), verbose=False)
    assert set(out) == {"full (untruncated)", "rank=1", "rank=2", "rank=3"}
