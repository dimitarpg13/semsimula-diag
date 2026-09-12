"""Creation-gate temperature and register-level saturation.

Ported from Cells 6d-3 / 6d-4 (``probe_hot_rows``, ``probe_gate_saturation``,
``sweep_log_tau_history``). Background: Mitigations SS48/SS48.8 and
``Register_Temperature_Instability_in_the_Fock_Creation_Gate.md``.

This family is distinct from the V_theta/curvature probes: it looks at the
creation gate's per-register temperature, where a single cold register can
take essentially all of ``log_tau``'s gradient.

**Read across REGISTERS, not rows.** SS49.4 is the cautionary tale: the
original test maxed the scaled score over all registers before comparing
rows, so one dominant register made the statistic near-constant and it could
not have separated rows whatever the true mechanism was.
"""
from __future__ import annotations

import re as _re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from ..report import ProbeResult
from .context import ProbeContext

__all__ = ["sweep_log_tau_history", "probe_gate_saturation"]

_LOG_TAU_KEY = "creation_gate_qkv.log_tau"
_REGISTER_KEY = "register_embed"


def _tau_row(state_dict, step: int, kind: str, focus: int) -> Optional[Dict[str, Any]]:
    lt = state_dict.get(_LOG_TAU_KEY)
    if lt is None:
        return None
    lt = lt.float()
    tau = lt.exp().clamp(min=1e-4)
    emb = state_dict.get(_REGISTER_KEY)
    focus = min(focus, tau.numel() - 1)
    return {
        "step": step, "kind": kind,
        "log_tau_focus": float(lt[focus]),
        "tau_focus": float(tau[focus]),
        "tau_min": float(tau.min()),
        "tau_argmin": int(tau.argmin()),
        "tau_median": float(tau.median()),
        "tau_max": float(tau.max()),
        # rank 1 = hottest; n = coldest
        "tau_focus_rank": 1 + int((tau > tau[focus]).sum()),
        "n_registers": int(tau.numel()),
        "emb_focus_norm": (float(emb[focus].float().norm())
                           if emb is not None else float("nan")),
        "emb_median_norm": (float(emb.float().norm(dim=-1).median())
                            if emb is not None else float("nan")),
    }


def sweep_log_tau_history(ctx: ProbeContext, focus: int = 14,
                          extra_checkpoints: Sequence[Path] = (),
                          suffix: str = "spikebatch",
                          verbose: bool = True) -> ProbeResult:
    """Temperature trajectory mined from every bundle already on disk.

    Reads ``model_state_dict`` out of the capture bundles rather than real
    checkpoints -- bundles carry weights but no optimizer state, so this is
    far cheaper than a checkpoint sweep, and needs no forward pass at all.

    Use it to answer "is this register drifting, and since when?" without
    paying for a single replay.

    Args:
        focus: the register to track. The default follows the notebook's
            ``FOCUS_REGISTER``; it only selects which register gets its own
            columns -- the pool-wide min/median/max are always reported, so
            check ``tau_argmin`` rather than assuming the focus is still the
            interesting one.
    """
    ctx.require("store")
    store = ctx.store
    found: List[tuple] = []
    for directory in (store.ckpt_dir, store._archive_dir(suffix)):
        if not directory.exists():
            continue
        for p in sorted(directory.glob(f"{store.ckpt_prefix}_step*_{suffix}.pt")):
            m = _re.search(rf"_step(\d+)_{suffix}\.pt$", p.name)
            if m:
                found.append((int(m.group(1)), p, suffix))
    for p in extra_checkpoints:
        p = Path(p)
        m = _re.search(r"_step(\d+)", p.name)
        found.append((int(m.group(1)) if m else -1, p, "checkpoint"))

    seen, rows = set(), []
    for step, p, kind in sorted(found):
        if step in seen:            # same bundle in live dir and archive
            continue
        seen.add(step)
        try:
            b = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [skip] {p.name}: {type(e).__name__}: {e}")
            continue
        row = _tau_row(b.get("model_state_dict", b), step, kind, focus)
        if row is not None:
            rows.append(row)

    if verbose and rows:
        print(f'{"step":>8} {"tau_focus":>10} {"rank":>6} {"tau_min":>9} '
              f'{"argmin":>7} {"tau_med":>9} {"emb_focus":>10}')
        for r in rows:
            print(f'{r["step"]:8d} {r["tau_focus"]:10.4f} '
                  f'{r["tau_focus_rank"]:3d}/{r["n_registers"]:<2d} '
                  f'{r["tau_min"]:9.4f} {r["tau_argmin"]:7d} '
                  f'{r["tau_median"]:9.4f} {r["emb_focus_norm"]:10.4f}')
    elif verbose:
        print(f"  no {suffix} bundles found under {store.ckpt_dir} "
              f"or {store._archive_dir(suffix)}")

    drift = ((rows[-1]["tau_focus"] - rows[0]["tau_focus"]) if len(rows) > 1
             else 0.0)
    return ProbeResult(
        probe_name="log_tau_history",
        metrics={"n_snapshots": len(rows), "focus": focus,
                 "tau_focus_drift": drift},
        raw={"rows": rows},
    )


def probe_gate_saturation(ctx: ProbeContext, step_tag: int,
                          focus: int = 14, clamp: Optional[float] = None,
                          verbose: bool = True) -> ProbeResult:
    """Per-register creation-gate temperature at one capture.

    Weight-only: reads ``log_tau`` straight out of the bundle, so it needs no
    forward pass and works against any checkpoint kind.

    ``focus`` only highlights one register in the output; the full ranking is
    always computed. **Do not assume the focus register is still the
    interesting one** -- read ``tau_argmin``.

    Note: the notebook's version additionally replays a microbatch to measure
    per-register x per-layer readout clamp occupancy and salience. That part
    is not ported yet; what is here is the temperature ranking that the
    occupancy analysis is read against.
    """
    ctx.require("store")
    bundle, path = ctx.store.load(step_tag)
    sd = bundle.get("model_state_dict", bundle)
    lt = sd.get(_LOG_TAU_KEY)
    if lt is None:
        raise RuntimeError(
            f"{_LOG_TAU_KEY} not in the bundle's state dict; this model has "
            f"no per-register creation-gate temperature.")
    lt = lt.float()
    tau = lt.exp().clamp(min=1e-4)
    order = torch.argsort(tau)              # ascending: coldest first
    focus = min(focus, tau.numel() - 1)

    if verbose:
        print(f'[sat] step {bundle.get("step", step_tag)}: '
              f'{tau.numel()} registers')
        print(f'{"rank":>5} {"register":>9} {"tau":>10}')
        shown = list(order[:8].tolist())
        if focus not in shown:
            shown.append(focus)
        for rank, k in enumerate(shown, start=1):
            mark = "  <-- focus" if k == focus else ""
            r = 1 + int((tau < tau[k]).sum())
            print(f"{r:5d} {k:9d} {float(tau[k]):10.4f}{mark}")
        print(f'[sat] coldest register {int(tau.argmin())} '
              f'(tau={float(tau.min()):.4f}) vs median '
              f'{float(tau.median()):.4f}')

    return ProbeResult(
        probe_name="gate_saturation",
        step_tag=bundle.get("step", step_tag),
        metrics={
            "tau_min": float(tau.min()), "tau_argmin": int(tau.argmin()),
            "tau_median": float(tau.median()), "tau_max": float(tau.max()),
            "tau_focus": float(tau[focus]),
            "tau_focus_rank": 1 + int((tau < tau[focus]).sum()),
            "n_registers": int(tau.numel()),
        },
        raw={"tau": [float(v) for v in tau],
             "coldest_order": [int(v) for v in order]},
    )
