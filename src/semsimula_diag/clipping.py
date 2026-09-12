"""Per-parameter-group gradient clipping, and both clip strategies.

Ported from ``grad_clip_utils.py`` and the inline ``clip_then_sum`` splice in
Cell 6 / Cell 6d of
``colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb``
(semsimula-paper repo).

Corresponds to ``semsimula_diag.clipping`` in
``companion_notes/Diagnostic_Programme_in_CfC_BAOAB_Integrator.md`` SS11.2-SS11.3,
covering migration steps 1 and 7 of SS11.6. Step 1 (the four functions below
originally extracted as ``grad_clip_utils.py``) was already a standalone
module; step 7 folds in the ``clip_then_sum`` splice, which until now lived
inline in Cell 6 with a second, hand-mirrored copy in Cell 6d.

**The one behavioural change from the notebook**: the notebook's ``_cts_*``
helpers read ``CLIP_THEN_SUM_GROUPS`` / ``CLIP_THEN_SUM_THRESHOLD`` /
``_GRAD_CLIP_CFG`` / ``PER_GROUP_CLIP`` out of ``globals()``. Here those are
fields on :class:`GradClipConfig`, passed explicitly. The no-op fallback is
preserved exactly: an unset/empty ``clip_then_sum_groups`` makes
:func:`clip_then_sum_params` return ``{}``, and every downstream call becomes
a true no-op -- which is what keeps this safe against pre-SS45.4 captures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple

import torch
import torch.nn as nn

__all__ = [
    "GradClipConfig",
    "assign_clip_group",
    "per_group_grad_norms",
    "clip_grads_per_group",
    "clip_then_sum_params",
    "ClipThenSum",
]


@dataclass(frozen=True)
class GradClipConfig:
    """Bundles the knobs that determine how a parameter's name maps to a clip
    group and threshold, so callers don't have to pass them separately at
    every call site.

    Attributes:
        default_clip: clip threshold for any parameter that doesn't match an
            entry in `overrides`.
        overrides: substring (case-insensitive) -> clip threshold. Any
            override whose key appears in a parameter's name wins, and the
            parameter is placed in a group named f'override:{key}'.

            ORDER MATTERS: `assign_clip_group` returns the FIRST substring
            hit while iterating this dict. A more specific key must come
            before a less specific one that also matches -- e.g. 'log_tau'
            must precede 'creation_gate', since
            'creation_gate_qkv.log_tau' matches both, and putting it later
            would make it dead config that silently never fires
            (Mitigations SS49.8).
        watchdog_exclude_groups: group keys (post-override, i.e. already in
            'override:...' form where applicable) to exclude from the
            *aggregate* norm returned by `clip_grads_per_group` -- their own
            per-group norm is still computed and clipped normally, they're
            just not counted toward the watchdog's total.
        clip_then_sum_groups: group keys (same post-override form) whose
            gradients are clipped PER MICROBATCH and then summed, instead of
            summed across microbatches and then clipped once. Empty (the
            default) reproduces pre-SS45.4 `sum_then_clip` behaviour exactly.
        clip_then_sum_threshold: joint clip threshold applied to each
            clip_then_sum group's own per-microbatch contribution. Required
            whenever `clip_then_sum_groups` is non-empty.
    """

    default_clip: float
    overrides: Dict[str, float] = field(default_factory=dict)
    watchdog_exclude_groups: FrozenSet[str] = field(default_factory=frozenset)
    clip_then_sum_groups: FrozenSet[str] = field(default_factory=frozenset)
    clip_then_sum_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.clip_then_sum_groups and self.clip_then_sum_threshold is None:
            raise ValueError(
                "clip_then_sum_threshold is required when "
                "clip_then_sum_groups is non-empty; without it "
                "clip_grad_norm_ would be called with max_norm=None"
            )


def assign_clip_group(pname: str, cfg: GradClipConfig) -> Tuple[str, float]:
    """Map a parameter name to (clip-group key, clip threshold)."""
    low = pname.lower()
    for sub, thr in cfg.overrides.items():
        if sub.lower() in low:
            return f'override:{sub}', thr
    return pname.split('.', 1)[0], cfg.default_clip


def per_group_grad_norms(mdl: nn.Module, cfg: GradClipConfig) -> Dict[str, float]:
    """Read-only per-clip-group L2 grad norm (no clipping applied)."""
    groups: Dict[str, List[nn.Parameter]] = {}
    for n, p in mdl.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        key, _ = assign_clip_group(n, cfg)
        groups.setdefault(key, []).append(p)
    out: Dict[str, float] = {}
    for key, ps in groups.items():
        sq = 0.0
        for p in ps:
            sq += float(p.grad.detach().norm()) ** 2
        out[key] = sq ** 0.5
    return out


def clip_grads_per_group(
    mdl: nn.Module, cfg: GradClipConfig
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Clip each group's grads independently in place; return the aggregate
    pre-clip norm (excluding `cfg.watchdog_exclude_groups`) plus every
    group's own pre-clip norm (`nn.utils.clip_grad_norm_` returns the norm
    computed *before* it applies the rescale, for every group -- including
    excluded ones, which are just left out of the aggregate).
    """
    groups: Dict[str, List[nn.Parameter]] = {}
    thr: Dict[str, float] = {}
    device = None
    for n, p in mdl.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if device is None:
            device = p.grad.device
        key, mx = assign_clip_group(n, cfg)
        groups.setdefault(key, []).append(p)
        thr[key] = mx
    total_sq = torch.zeros((), device=device) if device is not None else torch.zeros(())
    per_group: Dict[str, float] = {}
    for key, ps in groups.items():
        gn = nn.utils.clip_grad_norm_(ps, thr[key])
        per_group[key] = float(gn)
        if key not in cfg.watchdog_exclude_groups:
            total_sq = total_sq + gn.detach() ** 2
    return total_sq.sqrt(), per_group


# ---------------------------------------------------------------------------
# clip_then_sum (Mitigations SS45.3-SS45.4)
# ---------------------------------------------------------------------------

def clip_then_sum_params(
    mdl: nn.Module, cfg: GradClipConfig
) -> Dict[str, List[nn.Parameter]]:
    """``{group_key: [nn.Parameter, ...]}`` for whichever groups this run
    applies clip_then_sum to, resolved fresh against ``mdl`` via the same
    :func:`assign_clip_group` the watchdog and optimizer use.

    Returns ``{}`` -- a true no-op downstream -- when clip_then_sum is not
    configured, which keeps every caller safe against pre-SS45.4 notebooks
    and bundles.
    """
    if not cfg.clip_then_sum_groups:
        return {}
    out: Dict[str, List[nn.Parameter]] = {}
    for n, p in mdl.named_parameters():
        if not p.requires_grad:
            continue
        key, _ = assign_clip_group(n, cfg)
        if key in cfg.clip_then_sum_groups:
            out.setdefault(key, []).append(p)
    return out


class ClipThenSum:
    """Per-microbatch clip-then-sum accumulator.

    Replaces the notebook's ``_cts_group_params`` / ``_cts_apply_microbatch``
    / ``_cts_splice_back`` trio plus the caller-held ``_cts_running`` dict
    with one object, while preserving the mechanics exactly: the same
    ``nn.utils.clip_grad_norm_`` call, the same running-total-keyed-by
    ``id(param)`` accumulation, and the same post-loop splice into ``.grad``.

    Usage mirrors the live training loop and the replay helpers alike::

        cts = ClipThenSum(model, cfg)
        for xb, yb in microbatches:
            loss = ...
            (loss / grad_accum).backward()
            cts.apply_microbatch()      # clip this microbatch's own share
        cts.splice_back()               # install the accumulated total

    A no-op throughout when ``cfg.clip_then_sum_groups`` is empty, so it is
    safe to wrap unconditionally around any accumulation loop.
    """

    def __init__(self, mdl: nn.Module, cfg: GradClipConfig) -> None:
        self.cfg = cfg
        self.params = clip_then_sum_params(mdl, cfg)
        self.running: Dict[int, torch.Tensor] = {}

    @property
    def active(self) -> bool:
        return bool(self.params)

    def apply_microbatch(self) -> None:
        """After ONE microbatch's ``backward()``: jointly clip each
        clip_then_sum group's own contribution, fold it into the running
        total, then zero ``.grad`` so the caller's normal accumulation never
        double-counts it.
        """
        threshold = self.cfg.clip_then_sum_threshold
        for gparams in self.params.values():
            if not any(p.grad is not None for p in gparams):
                continue
            nn.utils.clip_grad_norm_(gparams, threshold)
            for p in gparams:
                if p.grad is None:
                    continue
                pid = id(p)
                contrib = p.grad.detach().clone()
                self.running[pid] = (
                    self.running[pid] + contrib if pid in self.running else contrib
                )
                p.grad.zero_()

    def splice_back(self) -> None:
        """After the full accumulation loop: replace each clip_then_sum
        group's (already-zeroed) ``.grad`` with its accumulated,
        per-microbatch-clipped running total, so every downstream reader sees
        exactly what the live optimizer step saw.
        """
        for gparams in self.params.values():
            for p in gparams:
                pid = id(p)
                if pid in self.running:
                    p.grad = self.running[pid]
