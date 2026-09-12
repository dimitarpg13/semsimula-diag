"""Tests against real capture bundles from the live 100K run.

These exercise BundleStore on genuine files rather than synthetic ones, and
pin the two facts the post-100K analysis depends on (Post_100K checklist
SS1/SS2.6): the recorded pre-clip norms, and the register/V_theta ratios that
separate step 87196 from step 90360.

Skipped automatically when the bundles are absent -- see conftest.py.
"""
from __future__ import annotations

import pytest

from conftest import requires_bundles  # pytest puts tests/ on sys.path

from semsimula_diag import BundleStore

# `bundle_dir` and `ckpt_prefix` come from conftest.py automatically --
# fixtures need no import.
pytestmark = requires_bundles


@pytest.fixture
def store(bundle_dir, ckpt_prefix):
    # the bundles sit flat in one directory, so point both the live dir and
    # the archive root at it
    return BundleStore(ckpt_dir=bundle_dir, ckpt_prefix=ckpt_prefix,
                       archive_root=bundle_dir, verbose=False)


def test_resolves_and_loads_real_bundle(store):
    path, src = store.resolve(87196)
    assert src == 'live' and path.exists()
    bundle, loaded = store.load(87196)
    assert loaded == path
    assert bundle['step'] == 87196


def test_bundle_shape_matches_what_probes_expect(store):
    bundle, _ = store.load(87196)
    assert set(bundle) >= {
        'batches', 'model_state_dict', 'rng_state_cpu', 'rng_state_cuda',
        'step', 'grad_accum', 'pre_clip_grad_norm', 'top_groups',
    }
    assert bundle['grad_accum'] == 4
    assert len(bundle['batches']) == bundle['grad_accum']
    assert len(bundle['model_state_dict']) == 152

    x, y = bundle['batches'][0]
    assert x.shape == y.shape == (8, 512)
    # Batches are stored as NUMPY arrays, exactly as get_batch() produced
    # them -- not tensors. Any probe must convert before the forward pass
    # (the live loop does torch.from_numpy(...).to(DEVICE)); feeding them
    # straight to nn.Embedding raises
    # "argument 'indices' must be Tensor, not numpy.ndarray".
    import numpy as np
    assert isinstance(x, np.ndarray), type(x)
    assert isinstance(y, np.ndarray), type(y)


def test_rng_state_stays_cpu_bytetensor(store):
    """map_location='cpu' is load-bearing: torch.set_rng_state rejects
    anything but a CPU ByteTensor, so a bundle loaded onto the training
    device cannot be replayed deterministically."""
    import torch
    bundle, _ = store.load(87196)
    rng = bundle['rng_state_cpu']
    assert rng.dtype == torch.uint8
    assert rng.device.type == 'cpu'


@pytest.mark.parametrize('step_tag,pre_clip', [(87196, 2539.2), (90360, 567.26)])
def test_recorded_pre_clip_norms(store, step_tag, pre_clip):
    bundle, _ = store.load(step_tag)
    assert bundle['pre_clip_grad_norm'] == pytest.approx(pre_clip, abs=0.01)


@pytest.mark.parametrize('step_tag,ratio', [(87196, 8.59), (90360, 0.63)])
def test_register_vtheta_ratio_from_bundle(store, step_tag, ratio):
    """The SS2.6 discriminator, read straight off the bundle.

    87196 is the register-decoupled tail event; 90360 is V_theta-slaved.
    """
    tg = store.load(step_tag)[0]['top_groups']
    assert tg['override:register'] / tg['V_theta'] == pytest.approx(ratio, abs=0.01)
