"""Temperature probes -- weight-only, so fully verifiable on CPU.

Run against the real bundles when available (see conftest), otherwise
against a synthetic state dict with a known coldest register.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import CKPT_PREFIX, have_bundle

from semsimula_diag import BundleStore, ProbeContext
from semsimula_diag.probes import tau_saturation

LOG_TAU = "creation_gate_qkv.log_tau"


@pytest.fixture
def synth_store(tmp_path):
    """register 3 is deliberately the coldest; 14 (the historical focus) is not."""
    import math
    log_tau = torch.full((16,), math.log(6.5))
    log_tau[3] = math.log(2.0)          # coldest
    log_tau[14] = math.log(6.0)         # cool-ish, but NOT the minimum
    ckpt = tmp_path / "c"; ckpt.mkdir()
    for step in (100, 200):
        torch.save({"step": step,
                    "model_state_dict": {LOG_TAU: log_tau,
                                         "register_embed": torch.ones(16, 4)}},
                   ckpt / f"run_step{step}_spikebatch.pt")
    return BundleStore(ckpt_dir=ckpt, ckpt_prefix="run", archive_root=tmp_path,
                       verbose=False)


@pytest.fixture
def ctx(synth_store):
    return ProbeContext(model=nn.Linear(2, 2), store=synth_store)


def test_gate_saturation_finds_the_coldest_register(ctx):
    r = tau_saturation.probe_gate_saturation(ctx, 100, verbose=False)
    assert r.metrics["tau_argmin"] == 3
    assert r.metrics["tau_min"] == pytest.approx(2.0, rel=1e-4)
    assert r.metrics["n_registers"] == 16


def test_focus_does_not_restrict_the_search(ctx):
    """SS49.4's lesson: `focus` only highlights one register in the output.
    The argmin must be found regardless of what focus is set to."""
    for focus in (0, 3, 14):
        r = tau_saturation.probe_gate_saturation(ctx, 100, focus=focus,
                                                 verbose=False)
        assert r.metrics["tau_argmin"] == 3, f"focus={focus} changed the argmin"


def test_focus_rank_is_reported_so_you_can_see_it_is_not_the_coldest(ctx):
    r = tau_saturation.probe_gate_saturation(ctx, 100, focus=14, verbose=False)
    assert r.metrics["tau_focus"] == pytest.approx(6.0, rel=1e-4)
    assert r.metrics["tau_focus_rank"] > 1, "focus should not rank as coldest"


def test_missing_log_tau_raises_rather_than_returning_empty(tmp_path):
    ckpt = tmp_path / "c"; ckpt.mkdir()
    torch.save({"step": 1, "model_state_dict": {"other": torch.ones(3)}},
               ckpt / "run_step1_spikebatch.pt")
    store = BundleStore(ckpt_dir=ckpt, ckpt_prefix="run", verbose=False)
    ctx = ProbeContext(model=nn.Linear(2, 2), store=store)
    with pytest.raises(RuntimeError, match="log_tau"):
        tau_saturation.probe_gate_saturation(ctx, 1, verbose=False)


def test_history_sweep_finds_every_bundle_on_disk(ctx):
    r = tau_saturation.sweep_log_tau_history(ctx, verbose=False)
    assert r.metrics["n_snapshots"] == 2
    assert [row["step"] for row in r.raw["rows"]] == [100, 200]


def test_history_sweep_deduplicates_live_and_archive_copies(tmp_path):
    """The same step can exist in both the live ring and the archive."""
    import math
    ckpt = tmp_path / "c"; ckpt.mkdir()
    arch = tmp_path / "spikebatch_archive"; arch.mkdir()
    payload = {"step": 500,
               "model_state_dict": {LOG_TAU: torch.full((4,), math.log(5.0))}}
    torch.save(payload, ckpt / "run_step500_spikebatch.pt")
    torch.save(payload, arch / "run_step500_spikebatch.pt")
    store = BundleStore(ckpt_dir=ckpt, ckpt_prefix="run", archive_root=tmp_path,
                        verbose=False)
    ctx = ProbeContext(model=nn.Linear(2, 2), store=store)
    r = tau_saturation.sweep_log_tau_history(ctx, verbose=False)
    assert r.metrics["n_snapshots"] == 1, "step 500 counted twice"


@pytest.mark.skipif(not have_bundle(87196), reason="real bundle not present")
def test_against_the_real_step_87196_bundle(bundle_dir):
    """The register-temperature reading at a real capture."""
    store = BundleStore(ckpt_dir=bundle_dir, ckpt_prefix=CKPT_PREFIX,
                        archive_root=bundle_dir, verbose=False)
    ctx = ProbeContext(model=nn.Linear(2, 2), store=store)
    r = tau_saturation.probe_gate_saturation(ctx, 87196, verbose=False)
    assert r.metrics["n_registers"] == 32
    # the live log reported tau_min=5.22 @ register 14 around this step
    assert r.metrics["tau_min"] == pytest.approx(5.22, abs=0.05)
    assert r.metrics["tau_argmin"] == 14
