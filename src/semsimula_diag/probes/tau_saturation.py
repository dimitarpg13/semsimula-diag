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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from ..report import ProbeResult
from ._engine import iter_isolated_rows, patched_attrs, replayed
from .context import ProbeContext

__all__ = ["sweep_log_tau_history", "probe_gate_saturation", "probe_hot_rows"]

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


def probe_hot_rows(
    ctx: ProbeContext,
    step_tag: int,
    hot_rows: Sequence[Tuple[int, int]],
    track: Sequence[str] = ("creation_gate_qkv.log_tau",
                            "reverse_channel_scale"),
    readout_module: Any = None,
    readout_fn_names: Sequence[str] = ("_prefix_causal_creation_readout",
                                       "_causal_creation_readout"),
    readout_clamp: float = 40.0,
    verbose: bool = True,
) -> ProbeResult:
    """Per-register / per-layer element breakdown, plus the creation gate's
    pre-softmax score magnitudes -- full-batch and per-row.

    Ported from Cell 6d-3's ``probe_hot_rows`` (Mitigations SS48 follow-up).

    ``log_tau`` is shape ``(M,)`` -- one entry per REGISTER -- and
    ``reverse_channel_scale`` is ``(n_gate,)`` -- one per LAYER. Their
    reported group norms aggregate over those axes, which this looks inside
    of: it dumps the per-element breakdown of the full-batch gradient, then
    replays every row in isolation to see which rows drive which elements,
    alongside the creation gate's pre-softmax score magnitude -- captured by
    patching the module-level readout function, so no Q/K math is duplicated
    here.

    This is the probe behind the SS48.8 test: is a gradient-hot row a SCORE
    outlier (its creation-gate scores are unusually large), or only a
    backward-signal outlier (large gradient, ordinary scores)? The two imply
    different mechanisms.

    Args:
        hot_rows: ``[(microbatch, row), ...]`` to report in detail -- e.g.
            the rows :func:`~semsimula_diag.probes.row_attribution.attribute_spike_rows`
            already implicated. Every row is still replayed to establish
            rank; this only controls what gets printed and returned in
            detail.
        track: parameter names to break down per-element. The default
            follows the notebook; pass names as they appear in
            ``model.named_parameters()``.
        readout_module: the module object exposing the creation-gate readout
            functions -- ``model_fock_parf_v2`` in the training notebook,
            imported there as ``_mfv2``. **Required** -- there is no default,
            because guessing wrong here would silently record nothing rather
            than raise.
        readout_fn_names: names of the module-level functions to patch. Both
            must exist on ``readout_module``, or this raises immediately
            (via :func:`~semsimula_diag.probes._engine.patched_attrs`)
            rather than silently hooking nothing.
        readout_clamp: the readout's constant score-stabilising shift
            (``READOUT_CLAMP`` in the notebook). Score magnitudes are
            reported as a fraction of this, so it should match the live
            run's own value.

    Raises:
        ValueError: if ``readout_module`` is not supplied.
        AttributeError: if ``readout_module`` lacks either name in
            ``readout_fn_names``.
    """
    if readout_module is None:
        raise ValueError(
            "probe_hot_rows needs readout_module -- the module object "
            "exposing the creation-gate readout functions (e.g. "
            "model_fock_parf_v2, imported as _mfv2 in the training "
            "notebook). There is no default: guessing wrong would "
            "silently capture nothing rather than raise."
        )
    ctx.require("store", "clip_cfg", "forward_fn")
    bundle, path = ctx.store.load(step_tag)
    if verbose:
        print(f'[probe] loaded {path.name}  step={bundle["step"]}  '
              f'pre_clip_grad_norm={bundle.get("pre_clip_grad_norm")}')

    model = ctx.model
    name2param = dict(model.named_parameters())
    targets = [n for n in track if n in name2param]
    missing = [n for n in track if n not in name2param]
    if missing and verbose:
        print(f"[probe][WARN] not in named_parameters(), skipping: {missing}")

    score_calls: List[Dict[str, float]] = []

    def _make_wrapper(original):
        def _patched(scores, V, *a, **kw):
            with torch.no_grad():
                s = scores.detach().float()
                score_calls.append({"max_abs": float(s.abs().max()),
                                    "mean": float(s.mean()),
                                    "std": float(s.std())})
            return original(scores, V, *a, **kw)
        return _patched

    patches = {name: _make_wrapper for name in readout_fn_names}

    full_grads: Dict[str, Optional[torch.Tensor]] = {}
    row_recs: List[Dict[str, Any]] = []

    with patched_attrs(readout_module, patches):
        # Full-batch pass: reuses `replayed()` exactly as
        # `layer_profile.replay_spike_batch` does (weights, RNG, clip_then_sum
        # all handled identically). Its restoration does not fire until this
        # whole `with` block exits, so the per-row pass below still sees the
        # bundle's loaded weights.
        with replayed(ctx, bundle) as info:
            for n in targets:
                p = name2param[n]
                full_grads[n] = (p.grad.detach().float().reshape(-1).cpu().clone()
                                if p.grad is not None else None)
            full_score_max = max((c["max_abs"] for c in score_calls),
                                 default=float("nan"))

            # Per-row pass: isolated backward per row, RNG reset each time
            # (iter_isolated_rows), deliberately NOT going through
            # clip_then_sum -- this measures each row's own raw signal, not
            # what the live optimizer step would have applied.
            for mb, row, x, y, n_rows in iter_isolated_rows(ctx, bundle):
                score_calls.clear()
                loss, loss_ntp, _v, _f = ctx.forward_fn(model, x, y, ctx)
                if ctx.register_repulsion and hasattr(model, "pop_repulsion_loss"):
                    # Drained, deliberately unused: pop_repulsion_loss()
                    # must be called once per forward regardless (it holds
                    # live graph references, per its own docstring), but a
                    # single-row repulsion reading is not what this probe
                    # measures -- matches the notebook exactly.
                    model.pop_repulsion_loss()
                (loss / (info.grad_accum * n_rows)).backward()
                rec: Dict[str, Any] = {
                    "microbatch": mb, "row": row, "ntp": float(loss_ntp),
                    "score_max_abs": max((c["max_abs"] for c in score_calls),
                                         default=float("nan")),
                    "score_std_max": max((c["std"] for c in score_calls),
                                         default=float("nan")),
                }
                for n in targets:
                    p = name2param[n]
                    rec[n] = (p.grad.detach().float().reshape(-1).cpu().clone()
                             if p.grad is not None else None)
                row_recs.append(rec)
    # weights, grads and RNG are restored now (replayed()'s exit)

    axis = {"creation_gate_qkv.log_tau": "register",
           "reverse_channel_scale": "layer"}

    if verbose:
        for n in targets:
            g = full_grads.get(n)
            if g is None:
                continue
            tot_sq = float((g ** 2).sum()) or 1.0
            print(f'\n[probe] {n}: per-{axis.get(n, "element")} breakdown of '
                  f'the full-batch gradient (norm={tot_sq ** 0.5:.2f}, '
                  f'{g.numel()} entries)')
            print(f'    {"idx":>4}  {"grad":>12}  {"share_of_norm^2":>16}')
            for i in torch.argsort(g.abs(), descending=True).tolist():
                print(f'    {i:4d}  {float(g[i]):12.4f}  '
                      f'{float(g[i]) ** 2 / tot_sq:16.4f}')

        print(f'\n[probe] creation-gate score magnitude, full batch: '
              f'max|scaled score|={full_score_max:.2f}  '
              f'({full_score_max / readout_clamp:.2f}x the readout\'s '
              f'constant shift of {readout_clamp:g})')
        if row_recs:
            scores = sorted(r["score_max_abs"] for r in row_recs)
            n = len(scores)
            print(f'[probe] per-row max|scaled score|: '
                  f'median={scores[n // 2]:.2f}  '
                  f'p90={scores[int(0.9 * n)]:.2f}  max={scores[-1]:.2f}')

        hot = set(hot_rows)
        print(f'\n[probe] gradient-hot rows vs. batch (SS48.8: is the hot '
              f'row a SCORE outlier, or only a backward-signal outlier?):')
        for r in sorted(row_recs, key=lambda r: -r["score_max_abs"]):
            key = (r["microbatch"], r["row"])
            if key not in hot:
                continue
            rank = 1 + sum(1 for x in row_recs
                          if x["score_max_abs"] > r["score_max_abs"])
            print(f'    mb={key[0]} row={key[1]}  ntp={r["ntp"]:.3f}  '
                  f'max|scaled score|={r["score_max_abs"]:.2f}  '
                  f'(rank {rank}/{len(row_recs)}, 1 = largest in batch)')
            for nm in targets:
                g = r.get(nm)
                if g is None:
                    continue
                top = int(torch.argmax(g.abs()))
                print(f'        {nm}: peak {axis.get(nm, "element")} {top} '
                      f'(grad={float(g[top]):.4f}, '
                      f'row norm={float(g.norm()):.4f})')

    hot_out = []
    for r in row_recs:
        key = (r["microbatch"], r["row"])
        if key in set(hot_rows):
            hot_out.append({k: (v.tolist() if torch.is_tensor(v) else v)
                            for k, v in r.items()})

    return ProbeResult(
        probe_name="hot_rows",
        step_tag=bundle.get("step", step_tag),
        metrics={
            "full_score_max": full_score_max,
            "full_score_max_ratio": full_score_max / readout_clamp,
            "n_rows": len(row_recs),
            "n_hot_rows_found": len(hot_out),
        },
        raw={
            "full_grads": {k: (v.tolist() if v is not None else None)
                          for k, v in full_grads.items()},
            "hot_rows": hot_out,
        },
    )
