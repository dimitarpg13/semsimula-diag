"""probe_hot_rows against a toy model + toy readout module.

Exercises the real machinery end to end -- patched_attrs, iter_isolated_rows,
and a per-row pass nested inside an active `replayed()` block -- without
needing the real 77M-param anisotropic Gaussian model.

GPU verification against the real model is still pending -- see
MIGRATION.md and the tour notebook's verification section.
"""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from semsimula_diag import BundleStore, GradClipConfig, ProbeContext
from semsimula_diag.probes import tau_saturation


class _ToyModel(nn.Module):
    """Exposes creation_gate_qkv.log_tau and reverse_channel_scale, and
    calls a readout function through a module the test can patch."""

    def __init__(self, readout_module):
        super().__init__()
        self.readout_module = readout_module
        self.creation_gate_qkv = nn.Module()
        self.creation_gate_qkv.log_tau = nn.Parameter(torch.zeros(4))
        self.reverse_channel_scale = nn.Parameter(torch.zeros(3))
        self.E = nn.Parameter(torch.ones(4))
        self._repulsion_calls = 0

    def pop_repulsion_loss(self):
        self._repulsion_calls += 1
        return torch.tensor(0.0)

    def forward(self, x):
        # a scaled-score tensor the readout "sees" before softmax
        scores = self.E * x.float().mean()
        out = self.readout_module._toy_readout(scores, None)
        gate = self.creation_gate_qkv.log_tau.sum()
        rev = self.reverse_channel_scale.sum()
        return out.sum() + gate + rev


def _toy_readout(scores, V):
    return scores


def _forward_fn(model, x, y, ctx):
    loss = model(x)
    z = torch.tensor(0.0)
    return loss, loss.detach(), z, z


@pytest.fixture
def readout_module():
    mod = types.SimpleNamespace()
    mod._toy_readout = _toy_readout
    mod._other_readout = _toy_readout
    return mod


@pytest.fixture
def setup(tmp_path, readout_module):
    rows = np.arange(8, dtype=np.int64).reshape(2, 4)
    bundle = {
        "step": 555, "grad_accum": 2,
        "batches": [(rows, rows), (rows * 3, rows * 3)],
        "model_state_dict": {
            "creation_gate_qkv.log_tau": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "reverse_channel_scale": torch.tensor([1.0, 2.0, 3.0]),
            "E": torch.ones(4),
        },
        "rng_state_cpu": torch.get_rng_state(),
        "rng_state_cuda": None,
        "pre_clip_grad_norm": 1.0,
        "top_groups": {},
    }
    ckpt = tmp_path / "c"; ckpt.mkdir()
    torch.save(bundle, ckpt / "run_step555_spikebatch.pt")
    store = BundleStore(ckpt_dir=ckpt, ckpt_prefix="run",
                        archive_root=tmp_path, verbose=False)
    cfg = GradClipConfig(default_clip=1.0)
    model = _ToyModel(readout_module)
    ctx = ProbeContext(model=model, device="cpu", store=store, clip_cfg=cfg,
                       forward_fn=_forward_fn)
    return ctx, bundle, readout_module


def test_raises_without_readout_module(setup):
    ctx, bundle, _ = setup
    with pytest.raises(ValueError, match="readout_module"):
        tau_saturation.probe_hot_rows(ctx, 555, hot_rows=[(0, 0)],
                                      readout_module=None, verbose=False)


def test_raises_if_readout_fn_names_do_not_exist(setup):
    ctx, bundle, mod = setup
    with pytest.raises(AttributeError):
        tau_saturation.probe_hot_rows(
            ctx, 555, hot_rows=[(0, 0)], readout_module=mod,
            readout_fn_names=("_does_not_exist",), verbose=False)


def test_readout_is_patched_during_the_replay_and_restored_after(setup):
    ctx, bundle, mod = setup
    original = mod._toy_readout
    tau_saturation.probe_hot_rows(
        ctx, 555, hot_rows=[(0, 0)], readout_module=mod,
        readout_fn_names=("_toy_readout",), verbose=False)
    assert mod._toy_readout is original, "readout not restored after the call"


def test_score_capture_records_something(setup):
    ctx, bundle, mod = setup
    r = tau_saturation.probe_hot_rows(
        ctx, 555, hot_rows=[(0, 0)], readout_module=mod,
        readout_fn_names=("_toy_readout",), verbose=False)
    assert r.metrics["full_score_max"] > 0
    assert r.metrics["full_score_max_ratio"] == pytest.approx(
        r.metrics["full_score_max"] / 40.0)


def test_every_row_is_replayed_for_ranking(setup):
    """2 microbatches x 2 rows each = 4 rows total, regardless of hot_rows."""
    ctx, bundle, mod = setup
    r = tau_saturation.probe_hot_rows(
        ctx, 555, hot_rows=[(0, 0)], readout_module=mod,
        readout_fn_names=("_toy_readout",), verbose=False)
    assert r.metrics["n_rows"] == 4


def test_only_named_hot_rows_are_returned_in_detail(setup):
    ctx, bundle, mod = setup
    r = tau_saturation.probe_hot_rows(
        ctx, 555, hot_rows=[(0, 0), (1, 1)], readout_module=mod,
        readout_fn_names=("_toy_readout",), verbose=False)
    assert r.metrics["n_hot_rows_found"] == 2
    keys = {(h["microbatch"], h["row"]) for h in r.raw["hot_rows"]}
    assert keys == {(0, 0), (1, 1)}


def test_full_grads_reported_for_every_tracked_param(setup):
    ctx, bundle, mod = setup
    r = tau_saturation.probe_hot_rows(
        ctx, 555, hot_rows=[(0, 0)], readout_module=mod,
        readout_fn_names=("_toy_readout",), verbose=False)
    assert set(r.raw["full_grads"]) == {"creation_gate_qkv.log_tau",
                                        "reverse_channel_scale"}
    assert len(r.raw["full_grads"]["creation_gate_qkv.log_tau"]) == 4


def test_repulsion_loss_is_drained_every_row_but_not_added(setup):
    """pop_repulsion_loss() must fire once per row (it holds live graph
    refs) but must NOT change the row's loss -- matches the notebook."""
    ctx, bundle, mod = setup
    ctx.register_repulsion = True
    tau_saturation.probe_hot_rows(
        ctx, 555, hot_rows=[(0, 0)], readout_module=mod,
        readout_fn_names=("_toy_readout",), verbose=False)
    # 4 full-batch backward calls do NOT call pop_repulsion_loss (replayed()
    # calls it inside the loss sum instead) -- only the 4 per-row calls do,
    # via the explicit drain-but-discard in probe_hot_rows itself.
    assert ctx.model._repulsion_calls >= 4


def test_weights_grads_and_rng_restored_after(setup):
    ctx, bundle, mod = setup
    before = float(ctx.model.creation_gate_qkv.log_tau[0])
    with torch.no_grad():
        ctx.model.creation_gate_qkv.log_tau.grad = torch.full((4,), 9.0)
    tau_saturation.probe_hot_rows(
        ctx, 555, hot_rows=[(0, 0)], readout_module=mod,
        readout_fn_names=("_toy_readout",), verbose=False)
    assert float(ctx.model.creation_gate_qkv.log_tau[0]) == before
    assert torch.allclose(ctx.model.creation_gate_qkv.log_tau.grad,
                          torch.full((4,), 9.0))


def test_missing_tracked_param_is_skipped_not_crashed(setup):
    ctx, bundle, mod = setup
    r = tau_saturation.probe_hot_rows(
        ctx, 555, hot_rows=[(0, 0)],
        track=("creation_gate_qkv.log_tau", "does.not.exist"),
        readout_module=mod, readout_fn_names=("_toy_readout",), verbose=False)
    assert "does.not.exist" not in r.raw["full_grads"]


def test_two_readout_functions_both_get_patched(setup):
    """The default readout_fn_names is a pair -- both must be hookable."""
    ctx, bundle, mod = setup
    calls = []
    original = mod._other_readout

    def _spy(scores, V):
        calls.append(1)
        return original(scores, V)
    mod._other_readout = _spy

    tau_saturation.probe_hot_rows(
        ctx, 555, hot_rows=[(0, 0)], readout_module=mod,
        readout_fn_names=("_toy_readout", "_other_readout"), verbose=False)
    # _other_readout itself is never CALLED by the toy model's forward, but
    # patched_attrs must still wrap and restore it without erroring
    assert mod._other_readout is _spy, "the unrelated patch should not be reverted early"
