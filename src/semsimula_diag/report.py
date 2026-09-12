"""One shared result shape for every probe.

Corresponds to ``semsimula_diag.report`` /
``companion_notes/Diagnostic_Programme_in_CfC_BAOAB_Integrator.md`` SS11.4.

Every probe in the notebook today returns a bespoke ``dict``/tuple and prints
its own ad hoc table. A single shared shape removes that duplication and gives
one code path to render, diff, or serialize any of them -- and makes the
SS8 mode classifier a pure function of a result rather than prose repeated at
every call site.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

__all__ = ["ProbeResult"]


@dataclass
class ProbeResult:
    """The common return type for every ``semsimula_diag.probes.*`` function.

    Attributes:
        probe_name: e.g. ``"layer_profile"``, ``"clip_order"``.
        step_tag: the ``*_spikebatch.pt`` step this ran against.
        fidelity_gap_pct: replay-vs-capture agreement, or ``None`` for probes
            that do not replay (e.g. phase0 log readers). The notebook treats
            gaps up to ~0.002% as bit-exact.
        metrics: scalar outputs, e.g. ``{"L0_hgrad": 0.169, "dc_ratio": 1.09}``.
        per_layer: layer-indexed series, when applicable.
        per_group: clip-group-indexed series, when applicable.
        raw: the full original dict/report, for backward compatibility with
            call sites that still expect the notebook's bespoke shape.
    """

    probe_name: str
    step_tag: Optional[int] = None
    fidelity_gap_pct: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    per_layer: Optional[Dict[int, float]] = None
    per_group: Optional[Dict[str, float]] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not include_raw:
            d.pop("raw", None)
        return d

    def to_json(self, include_raw: bool = False, **kwargs: Any) -> str:
        """Serialize for archiving next to a run's other diagnostic outputs.

        ``per_layer`` keys are ints, which JSON turns into strings; that is
        lossy on a round-trip, so :meth:`from_json` converts them back.
        """
        return json.dumps(self.to_dict(include_raw=include_raw), **kwargs)

    @classmethod
    def from_json(cls, s: str) -> "ProbeResult":
        d = json.loads(s)
        if d.get("per_layer"):
            d["per_layer"] = {int(k): v for k, v in d["per_layer"].items()}
        return cls(**d)
