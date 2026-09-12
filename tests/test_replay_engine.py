"""The shared replay engine, against a toy model.

The engine is where a mistake is most expensive: every replay probe inherits
it, and a wrong step changes the answer WITHOUT raising -- which is exactly
how clip_then_sum silently degraded to sum_then_clip on real hardware.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from semsimula_diag import BundleStore, GradClipConfig, ProbeContext
from semsimula_diag.probes import MissingContextError, replayed, restored_model_state


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Parameter(torch.ones(4))
        self.register_embed = nn.Parameter(torch.ones(4))

    def forward(self, x):
        return (self.E.sum() + self.register_embed.sum()) * x.float().mean()


def _forward_fn(model, x, y, ctx):
    loss = model(x)
    z = torch.tensor(0.0)
    return loss, loss.detach(), z, z


@pytest.fixture
def setup(tmp_path):
    rows = np.arange(8, dtype=np.int64).reshape(2, 4)
    bundle = {
        "step": 777, "grad_accum": 2,
        "batches": [(rows, rows), (rows * 2, rows * 2)],
        "model_state_dict": {"E": torch.full((4,), 3.0),
                             "register_embed": torch.full((4,), 5.0)},
        "rng_state_cpu": torch.get_rng_state(),
        "rng_state_cuda": None,
        "pre_clip_grad_norm": 1.0,
        "top_groups": {},
    }
    ckpt = tmp_path / "c"; ckpt.mkdir()
    torch.save(bundle, ckpt / "run_step777_spikebatch.pt")
    store = BundleStore(ckpt_dir=ckpt, ckpt_prefix="run",
                        archive_root=tmp_path, verbose=False)
    cfg = GradClipConfig(default_clip=1.0, overrides={"register": 0.3},
                         clip_then_sum_groups=frozenset({"E"}),
                         clip_then_sum_threshold=0.1)
    ctx = ProbeContext(model=_Toy(), device="cpu", store=store, clip_cfg=cfg,
                       forward_fn=_forward_fn)
    return ctx, bundle


def test_replay_requires_forward_fn_and_clip_cfg(setup):
    ctx, bundle = setup
    bare = ProbeContext(model=_Toy(), store=ctx.store)
    with pytest.raises(MissingContextError, match="forward_fn"):
        with replayed(bare, bundle):
            pass


def test_replay_loads_the_bundles_pinned_weights(setup):
    ctx, bundle = setup
    assert float(ctx.model.E[0]) == 1.0
    with replayed(ctx, bundle):
        assert float(ctx.model.E[0]) == 3.0, "bundle weights not loaded"


def test_weights_grads_and_mode_are_restored_afterwards(setup):
    ctx, bundle = setup
    with torch.no_grad():
        ctx.model.register_embed.grad = torch.full((4,), 9.0)
    ctx.model.eval()
    with replayed(ctx, bundle):
        pass
    assert float(ctx.model.E[0]) == 1.0, "weights not restored"
    assert torch.allclose(ctx.model.register_embed.grad,
                          torch.full((4,), 9.0)), "grads not restored"
    assert not ctx.model.training, "training mode not restored"


def test_restoration_happens_even_when_the_body_raises(setup):
    ctx, bundle = setup
    with pytest.raises(RuntimeError, match="probe blew up"):
        with replayed(ctx, bundle):
            raise RuntimeError("probe blew up")
    assert float(ctx.model.E[0]) == 1.0


def test_clip_then_sum_is_applied_during_replay(setup):
    """The bug that cost a round-trip on the A100: if this silently no-ops,
    E comes back raw and unclipped across every microbatch."""
    ctx, bundle = setup
    with replayed(ctx, bundle) as info:
        assert info.clip_then_sum_groups == ["E"]
        # two microbatches, each clipped to 0.1 -> at most 0.2
        assert float(ctx.model.E.grad.norm()) <= 0.2 + 1e-5


def test_replay_reports_grad_accum_and_step(setup):
    ctx, bundle = setup
    with replayed(ctx, bundle) as info:
        assert info.step == 777
        assert info.grad_accum == 2
        assert info.recorded_pre_clip == 1.0


def test_fidelity_gap_is_none_without_a_recorded_norm(setup):
    ctx, bundle = setup
    bundle = dict(bundle); bundle.pop("pre_clip_grad_norm")
    with replayed(ctx, bundle) as info:
        assert info.fidelity_gap_pct(ctx) is None


def test_on_microbatch_fires_once_per_microbatch(setup):
    ctx, bundle = setup
    seen = []
    with replayed(ctx, bundle, on_microbatch=seen.append):
        pass
    assert seen == [0, 1]


def test_restored_model_state_can_skip_components(setup):
    ctx, _ = setup
    with restored_model_state(ctx, weights=False, grads=True, rng=False):
        with torch.no_grad():
            ctx.model.E.fill_(42.0)
    assert float(ctx.model.E[0]) == 42.0, "weights=False should not restore"
