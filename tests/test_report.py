"""Tests for the shared ProbeResult shape (Diagnostic Programme SS11.4)."""
from __future__ import annotations

from semsimula_diag.report import ProbeResult


def test_defaults_are_not_shared_between_instances():
    a, b = ProbeResult('x'), ProbeResult('y')
    a.metrics['k'] = 1.0
    assert b.metrics == {}, "mutable default leaked across instances"


def test_to_dict_omits_raw_unless_asked():
    r = ProbeResult('layer_profile', step_tag=87196, raw={'big': [1, 2, 3]})
    assert 'raw' not in r.to_dict()
    assert r.to_dict(include_raw=True)['raw'] == {'big': [1, 2, 3]}


def test_json_roundtrip_restores_int_per_layer_keys():
    r = ProbeResult(
        'layer_profile', step_tag=87196, fidelity_gap_pct=0.0019,
        metrics={'L0_hgrad': 0.169},
        per_layer={0: 0.169, 7: 0.004},
        per_group={'V_theta': 252.6},
    )
    back = ProbeResult.from_json(r.to_json())
    assert back.per_layer == {0: 0.169, 7: 0.004}, \
        "JSON stringifies int keys; from_json must convert them back"
    assert back.step_tag == 87196
    assert back.metrics == {'L0_hgrad': 0.169}
