"""Diagnostic instrumentation for Fock-PARFLM / SemSimula training runs.

Extracted from the CfC/BAOAB training notebooks in the ``semsimula-paper``
repo, following the module layout in
``companion_notes/Diagnostic_Programme_in_CfC_BAOAB_Integrator.md`` SS11.

Modules land in the SS11.6 migration order (low-risk first); see MIGRATION.md
for what is ported so far and what still lives only in the notebook.
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import clipping, probes, replay, report
from .clipping import (
    ClipThenSum,
    GradClipConfig,
    assign_clip_group,
    clip_grads_per_group,
    clip_then_sum_params,
    per_group_grad_norms,
)
from .replay import BundleStore, grad_restore, grad_snapshot, isolated_grads
from .probes import MissingContextError, ProbeContext
from .report import ProbeResult

__all__ = [
    "__version__",
    "clipping",
    "replay",
    "report",
    "probes",
    "ProbeContext",
    "MissingContextError",
    "GradClipConfig",
    "assign_clip_group",
    "per_group_grad_norms",
    "clip_grads_per_group",
    "clip_then_sum_params",
    "ClipThenSum",
    "grad_snapshot",
    "grad_restore",
    "isolated_grads",
    "BundleStore",
    "ProbeResult",
]
