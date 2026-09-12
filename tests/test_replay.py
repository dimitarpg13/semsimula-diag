"""Tests for semsimula_diag.replay.

The bundle-resolution tests cover the live-then-archive fallback added
2026-09-11 after the live ring evicted a bundle every replay helper needed --
behaviour that had no test while it lived in Cell 6d.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from semsimula_diag.replay import (
    BundleStore,
    grad_restore,
    grad_snapshot,
    isolated_grads,
)


def _model():
    m = nn.Module()
    m.a = nn.Parameter(torch.zeros(3))
    m.b = nn.Parameter(torch.zeros(3))
    return m


# --- gradient isolation -----------------------------------------------------

def test_snapshot_restore_roundtrip_preserves_grads_and_nones():
    m = _model()
    with torch.no_grad():
        m.a.grad = torch.full_like(m.a, 2.0)
        m.b.grad = None                       # mixed: one set, one unset

    saved = grad_snapshot(m)
    with torch.no_grad():                     # a probe scribbles over both
        m.a.grad = torch.full_like(m.a, 99.0)
        m.b.grad = torch.full_like(m.b, 99.0)
    grad_restore(m, saved)

    assert torch.allclose(m.a.grad, torch.full_like(m.a, 2.0))
    assert m.b.grad is None, "a None grad must be restored as None, not zeros"


def test_snapshot_is_a_deep_copy_not_an_alias():
    m = _model()
    with torch.no_grad():
        m.a.grad = torch.full_like(m.a, 2.0)
    saved = grad_snapshot(m)
    with torch.no_grad():
        m.a.grad.mul_(10.0)                   # in-place on the live grad
    assert torch.allclose(saved['a'], torch.full_like(m.a, 2.0)), \
        "snapshot aliased the live tensor instead of cloning it"


def test_isolated_grads_restores_even_when_body_raises():
    m = _model()
    with torch.no_grad():
        m.a.grad = torch.full_like(m.a, 2.0)

    with pytest.raises(RuntimeError, match="probe blew up"):
        with isolated_grads(m):
            with torch.no_grad():
                m.a.grad = torch.full_like(m.a, 99.0)
            raise RuntimeError("probe blew up")

    assert torch.allclose(m.a.grad, torch.full_like(m.a, 2.0)), \
        "grads must be restored on the exception path too"


# --- bundle resolution ------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    ckpt = tmp_path / 'checkpoints'
    ckpt.mkdir()
    (tmp_path / 'spikebatch_archive').mkdir()
    return BundleStore(ckpt_dir=ckpt, ckpt_prefix='run', archive_root=tmp_path,
                       verbose=False)


def test_resolve_prefers_live_over_archive(store, tmp_path):
    live = store.ckpt_dir / 'run_step100_spikebatch.pt'
    arch = tmp_path / 'spikebatch_archive' / 'run_step100_spikebatch.pt'
    torch.save({'step': 100, 'where': 'live'}, live)
    torch.save({'step': 100, 'where': 'archive'}, arch)

    path, src = store.resolve(100)
    assert src == 'live' and path == live
    bundle, _ = store.load(100)
    assert bundle['where'] == 'live'


def test_resolve_falls_back_to_archive_after_ring_rotation(store, tmp_path):
    """The live ring rotated this bundle out; the archive copy must be found."""
    arch = tmp_path / 'spikebatch_archive' / 'run_step200_spikebatch.pt'
    torch.save({'step': 200, 'where': 'archive'}, arch)

    path, src = store.resolve(200)
    assert src == 'archive' and path == arch
    bundle, _ = store.load(200)
    assert bundle['where'] == 'archive'


def test_missing_bundle_error_names_both_locations(store):
    with pytest.raises(FileNotFoundError) as e:
        store.load(999)
    msg = str(e.value)
    assert 'checkpoints' in msg and 'spikebatch_archive' in msg, msg
    assert '999' in msg


def test_archive_root_defaults_to_ckpt_dir_parent(tmp_path):
    ckpt = tmp_path / 'checkpoints'
    ckpt.mkdir()
    (tmp_path / 'prereload_archive').mkdir()
    s = BundleStore(ckpt_dir=ckpt, ckpt_prefix='run', verbose=False)
    arch = tmp_path / 'prereload_archive' / 'run_step5_prereload.pt'
    torch.save({'step': 5}, arch)
    _, src = s.resolve(5, suffix='prereload')
    assert src == 'archive'
