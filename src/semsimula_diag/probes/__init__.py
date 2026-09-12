"""One module per instrument family (Diagnostic Programme SS11.3).

All notebook probe families are now ported. The replay-based ones
(``layer_profile``, ``row_attribution``, ``precision_cap``, ``clip_order``,
``integrator``) share the engine in ``_engine`` and need a
``ProbeContext.forward_fn``; the weight-only ones (``stiffness``,
``tau_saturation``) and ``tokens`` do not.

**GPU verification pending** for every replay-based probe -- see
MIGRATION.md and the tour notebook's verification section.
"""
from __future__ import annotations

from . import (clip_order, integrator, layer_profile, precision_cap,
               row_attribution, stiffness, tau_saturation, tokens)
from ._engine import ReplayInfo, replayed, restored_model_state
from .context import MissingContextError, ProbeContext

__all__ = [
    "ProbeContext", "MissingContextError",
    "replayed", "restored_model_state", "ReplayInfo",
    "clip_order", "integrator", "layer_profile", "precision_cap",
    "row_attribution", "stiffness", "tau_saturation", "tokens",
]
