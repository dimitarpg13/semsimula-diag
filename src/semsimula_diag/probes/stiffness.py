"""Curvature and stiffness of the anisotropic Gaussian V_theta.

Ported from Cells 6b / 6b-2 / 6b-3 / 6b-4 of
``colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb``:
``stiffness_report``, ``sigma_lr_report``, ``sigma_lr_spectrum_report``,
``bracket_precision_lr_max``, ``spectrum_across_checkpoints``.

Diagnostic Programme SS11.2 maps all five to ``probes.stiffness``; SS17 and
``Curvature_Diagnostics_and_Rank_Selection_for_Aniso_Gaussian_Vtheta.md``
explain what they measure and how the rank decision reads them.

None of these compute gradients -- they hook the forward pass, record the
*realised* ``B_k`` (post-``_bound_lowrank``, i.e. what the forward actually
used) or the harmonic term, and restore the original method in a ``finally``.
The model's weights and training mode are left exactly as found.
"""
from __future__ import annotations

import contextlib
import copy
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from ..report import ProbeResult
from ._engine import patched_attrs
from .context import ProbeContext

__all__ = [
    "stiffness_report",
    "sigma_lr_report",
    "sigma_lr_spectrum_report",
    "sigma_lr_spectrum_by_site",
    "bracket_precision_lr_max",
    "spectrum_across_checkpoints",
]

_QUANTILES = (0.5, 0.9, 0.99, 0.999)
# torch.quantile refuses inputs beyond ~16M elements, and L*B*T*d reaches
# that at d=384, L=16 with a large auto-probed batch. Keep the exact max,
# subsample for the quantiles -- the tails are what matter.
_QUANTILE_SUBSAMPLE = 4_000_000


def _require_aniso(model, attr: str) -> None:
    if not hasattr(model.V_theta, attr):
        raise RuntimeError(
            f"V_theta has no {attr}(); need the anisotropic Gaussian family."
        )


@contextlib.contextmanager
def _patched(obj: Any, attr: str, replacement) -> Iterator[None]:
    """Swap a bound method, guaranteeing restoration."""
    original = getattr(obj, attr)
    setattr(obj, attr, replacement)
    try:
        yield original
    finally:
        setattr(obj, attr, original)


@contextlib.contextmanager
def _eval_mode(model) -> Iterator[None]:
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def _quantiles(t: torch.Tensor, qs: Sequence[float] = _QUANTILES
               ) -> Tuple[List[float], float]:
    """(quantiles, exact max), subsampling only for the quantile call."""
    exact_max = float(t.max())
    if t.numel() > _QUANTILE_SUBSAMPLE:
        t = t[torch.randint(0, int(t.numel()), (_QUANTILE_SUBSAMPLE,))]
    q = torch.tensor(qs, dtype=torch.float64)
    return [float(v) for v in torch.quantile(t.double(), q)], exact_max


def _record_context_components(ctx: ProbeContext, x: torch.Tensor,
                               collect) -> None:
    """Run one forward with ``context_components`` recording into ``collect``.

    ``collect(B)`` is called for each non-empty low-rank factor the forward
    actually used.
    """
    model = ctx.model
    _require_aniso(model, "context_components")
    original = model.V_theta.context_components

    def _recording(xis):
        comps = original(xis)
        # comps: list of (mu, a, w, B) per xi-channel
        for (_mu, _a, _w, B) in comps:
            if B.shape[-1] == 0:
                continue
            collect(B)
        return comps

    with _patched(model.V_theta, "context_components", _recording), \
            _eval_mode(model):
        with torch.enable_grad():
            model(x)


def stiffness_report(ctx: ProbeContext, x: torch.Tensor,
                     dt: Optional[float] = None) -> ProbeResult:
    """Distribution of ``omega*dt`` over layers, tokens and dimensions --
    the explicit-kick stability wall (Mitigations SS29).

    ``omega*dt >= 2`` is unstable, ``>= 1`` marginal.
    """
    model = ctx.model
    _require_aniso(model, "harmonic_terms")
    dt = float(model.cfg.dt if dt is None else dt)

    seen: List[torch.Tensor] = []
    original = model.V_theta.harmonic_terms

    def _recording(xis, h):
        k_diag, s = original(xis, h)
        seen.append(k_diag.detach().float().flatten().cpu())
        return k_diag, s

    # The wall is a property of the baoab_cfc explicit kick, so measure
    # under that integrator regardless of how the model is configured.
    saved = (model.cfg.integrator, model.cfg.vtheta_analytic_force)
    with _patched(model.V_theta, "harmonic_terms", _recording):
        model.cfg.integrator, model.cfg.vtheta_analytic_force = 'baoab_cfc', True
        try:
            with _eval_mode(model), torch.enable_grad():
                model(x)
        finally:
            model.cfg.integrator, model.cfg.vtheta_analytic_force = saved

    k = torch.cat(seen)
    mass = float(model.compute_mass(x).mean())
    wdt = (k.clamp(min=0) / mass).sqrt() * dt
    qs, wdt_max = _quantiles(wdt)
    return ProbeResult(
        probe_name="stiffness",
        metrics={
            "n_samples": int(wdt.numel()), "mean_mass": mass,
            "median": qs[0], "p90": qs[1], "p99": qs[2], "p999": qs[3],
            "max": wdt_max,
            "frac_unstable": float((wdt > 2.0).float().mean()),
            "frac_marginal": float((wdt > 1.0).float().mean()),
        },
    )


def sigma_lr_report(ctx: ProbeContext, x: torch.Tensor) -> ProbeResult:
    """Distribution of the raw ``sigma_max(B_k)^2`` -- the quantity
    ``precision_lr_max`` caps directly (SS3.3, Mitigations SS31.3).

    Deliberately NOT combined with ``a_k`` or ``g_k`` (unlike
    :func:`stiffness_report`'s ``omega*dt``): bracketing a
    ``precision_lr_max`` budget needs the raw per-well spectral norm, not a
    quantity already mixed with the bump weight or the diagonal precision.
    """
    seen: List[torch.Tensor] = []

    def _collect(B: torch.Tensor) -> None:
        # sigma_max(B_k)^2 from the SVD of B directly rather than
        # eigvalsh(B^T B): forming the Gram squares the condition number and
        # can make the symmetric-eigen driver fail to converge on degenerate
        # wells (same failure mode fixed in cfc_baoab.lowrank_modes).
        s_max_sq = torch.linalg.svdvals(B)[..., 0] ** 2
        seen.append(s_max_sq.detach().float().flatten().cpu())

    _record_context_components(ctx, x, _collect)
    if not seen:
        return ProbeResult(probe_name="sigma_lr", metrics={"n_samples": 0})

    s = torch.cat(seen)
    qs, s_max = _quantiles(s)
    return ProbeResult(
        probe_name="sigma_lr",
        metrics={
            "n_samples": int(s.numel()),
            "median": qs[0], "p90": qs[1], "p99": qs[2], "p999": qs[3],
            "max": s_max,
        },
    )


def sigma_lr_spectrum_report(ctx: ProbeContext, x: torch.Tensor) -> ProbeResult:
    """The FULL singular-value spectrum of ``B_k``, not just ``sigma_max``
    -- the effective-rank measurement behind the ANISO_RANK decision (SS17).

    Reports the participation ratio ``PR = (sum s_i^2)^2 / sum s_i^4``, which
    lives in ``[1, rank]``, alongside the Frobenius-norm distribution. Read
    ``fro_p50`` FIRST: the rank argument only applies once the Frobenius cap
    is actually binding, which is what makes rank a pure redistribution knob.
    """
    svals: List[torch.Tensor] = []

    def _collect(B: torch.Tensor) -> None:
        s = torch.linalg.svdvals(B)                 # (..., K, r), descending
        svals.append(s.detach().float().reshape(-1, s.shape[-1]).cpu())

    _record_context_components(ctx, x, _collect)
    if not svals:
        return ProbeResult(probe_name="sigma_lr_spectrum",
                           metrics={"n_samples": 0})

    s = torch.cat(svals).double()                   # (N, r)
    rank = s.shape[-1]
    s2 = s ** 2
    fro2 = s2.sum(-1)                               # ||B_k||_F^2
    pr = (fro2 ** 2) / (s2 ** 2).sum(-1).clamp(min=1e-300)  # guard dead wells
    spec = (s / s[:, :1].clamp(min=1e-300)).mean(0)  # mean sigma_i / sigma_1

    q = torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)
    pr_q, fro_q = torch.quantile(pr, q), torch.quantile(fro2.sqrt(), q)
    return ProbeResult(
        probe_name="sigma_lr_spectrum",
        metrics={
            "n_samples": int(s.shape[0]), "rank": rank,
            "pr_p05": float(pr_q[0]), "pr_p50": float(pr_q[1]),
            "pr_p95": float(pr_q[2]), "pr_mean": float(pr.mean()),
            "fro_p05": float(fro_q[0]), "fro_p50": float(fro_q[1]),
            "fro_p95": float(fro_q[2]),
            "sigma_max_sq_p50": float(torch.quantile(s2[:, 0], 0.5)),
        },
        raw={"spectrum": [float(v) for v in spec]},
    )


@contextlib.contextmanager
def _restored_weights(model) -> Iterator[None]:
    """Swap weights in and out without disturbing a resumable session.

    The same non-pollution invariant the notebook's ``bracket_precision_lr_max``
    and ``spectrum_across_checkpoints`` both hold: the live model's weights
    are restored in a ``finally``, so these are safe to run against a
    training session that will continue afterwards.
    """
    saved = copy.deepcopy(model.state_dict())
    try:
        yield
    finally:
        model.load_state_dict(saved)


def _load_weights_into(model, state_dict, device: str) -> None:
    model.load_state_dict(
        {k: v.to(device) for k, v in state_dict.items()}, strict=False)


def spectrum_across_checkpoints(
    ctx: ProbeContext,
    step_tags: Iterable[int] = (),
    include_prereload: Iterable[int] = (),
    best_ckpt_name: Optional[str] = None,
    n_batch: int = 4,
) -> Dict[str, ProbeResult]:
    """:func:`sigma_lr_spectrum_report` across best / spike / prereload
    checkpoints, on one fixed neutral batch (SS17).

    Answers both the rank question (is the budget saturated?) and the
    spectral-collapse question (does PR drop at a spike?) in a single pass.

    ``include_prereload`` takes ``_prereload.pt`` tags -- weights-only
    watchdog snapshots. They carry ``model_state_dict`` so they work here
    even though they can NOT be replayed (no batch, no RNG).
    """
    ctx.require("store", "batch_provider")
    x = ctx.neutral_batch(n_batch)
    reports: Dict[str, ProbeResult] = {}

    with _restored_weights(ctx.model):
        name = best_ckpt_name or f"{ctx.store.ckpt_prefix}_best.pt"
        best_path = ctx.store.ckpt_dir / name
        if best_path.exists():
            data = torch.load(best_path, map_location="cpu", weights_only=False)
            _load_weights_into(ctx.model, data["model_state_dict"], ctx.device)
            reports[f'BEST (step {data.get("step", "?")})'] = \
                sigma_lr_spectrum_report(ctx, x)
            del data

        for suffix, tags in (("spikebatch", step_tags),
                             ("prereload", include_prereload)):
            for tag in tags:
                path, src = ctx.store.resolve(tag, suffix)
                if path is None:
                    print(f"  [skip] no {suffix} bundle for step {tag}")
                    continue
                if src == "archive":
                    print(f"  [archive] step {tag} {suffix} -- evicted from "
                          f"the live ring, loaded from {suffix}_archive")
                data = torch.load(path, map_location="cpu", weights_only=False)
                _load_weights_into(ctx.model, data["model_state_dict"], ctx.device)
                res = sigma_lr_spectrum_report(ctx, x)
                res.step_tag = tag
                reports[f"{suffix} step {tag}"] = res
                del data
    return reports


def bracket_precision_lr_max(
    ctx: ProbeContext,
    step_tags: Iterable[int] = (),
    healthy_ckpt_name: Optional[str] = None,
    n_batch: int = 4,
) -> Dict[str, ProbeResult]:
    """Multi-checkpoint ``sigma_max(B_k)^2`` bracket, healthy vs spike
    regime (Mitigations SS42.4) -- what a ``precision_lr_max`` budget has to
    sit between.

    Same weight-restoring invariant as
    :func:`spectrum_across_checkpoints`.
    """
    ctx.require("store", "batch_provider")
    x = ctx.neutral_batch(n_batch)
    reports: Dict[str, ProbeResult] = {}

    with _restored_weights(ctx.model):
        name = healthy_ckpt_name or f"{ctx.store.ckpt_prefix}_best.pt"
        healthy = ctx.store.ckpt_dir / name
        if healthy.exists():
            data = torch.load(healthy, map_location="cpu", weights_only=False)
            _load_weights_into(ctx.model, data["model_state_dict"], ctx.device)
            reports[f'HEALTHY (step {data.get("step", "?")})'] = \
                sigma_lr_report(ctx, x)
            del data

        for tag in step_tags:
            path, src = ctx.store.resolve(tag, "spikebatch")
            if path is None:
                print(f"  [skip] no spikebatch bundle for step {tag}")
                continue
            data = torch.load(path, map_location="cpu", weights_only=False)
            _load_weights_into(ctx.model, data["model_state_dict"], ctx.device)
            res = sigma_lr_report(ctx, x)
            res.step_tag = tag
            reports[f"spike step {tag}"] = res
            del data
    return reports


def sigma_lr_spectrum_by_site(ctx: ProbeContext, x: torch.Tensor,
                              verbose: bool = True) -> ProbeResult:
    """Participation ratio broken out by (layer, channel), not pooled.

    :func:`sigma_lr_spectrum_report` reports global quantiles, which answer
    "is the rank budget used?" but not "is it used evenly?". Those are
    different decisions: a uniform PR justifies raising the single global
    rank, whereas a PR that is high in some banks and low in others argues
    for a per-bank allocation instead, since one global value then
    over-serves the saturated sites and under-serves the rest.

    The reported discriminator is a variance decomposition. ``between_std``
    is the spread of the per-site medians -- structure. ``within_std`` is
    the typical spread of wells inside a site -- noise. A ``between_frac``
    near 0 means the population is homogeneous and the pooled quantiles
    were telling the whole story; near 1 means the spread is organised by
    site and a global rank is the wrong instrument.

    Requires the model to expose ``_fock_layer_step``: without a true layer
    index every site would collapse into one, which is the failure mode
    this probe exists to avoid, so it raises rather than guessing.
    """
    model = ctx.model
    _require_aniso(model, "context_components")
    if not hasattr(model, "_fock_layer_step"):
        raise RuntimeError(
            "model exposes no _fock_layer_step, so the layer index cannot be "
            "recovered and every site would be recorded as layer 0. Use "
            "sigma_lr_spectrum_report for the pooled view instead.")

    sites: Dict[tuple, List[torch.Tensor]] = {}
    cur = {"layer": 0}

    def _wrap_layer(original):
        def _w(h, h_prev, r, salience, m_b, gamma, dt, layer_idx, *a, **kw):
            cur["layer"] = layer_idx
            return original(h, h_prev, r, salience, m_b, gamma, dt,
                            layer_idx, *a, **kw)
        return _w

    def _wrap_comps(original):
        def _w(xis):
            comps = original(xis)
            for ch, (_mu, _a, _w_, B) in enumerate(comps):
                if B.shape[-1] == 0:
                    continue
                sv = torch.linalg.svdvals(B.detach())
                sv = sv.float().reshape(-1, sv.shape[-1]).cpu().double()
                s2 = sv ** 2
                pr = (s2.sum(-1) ** 2) / (s2 ** 2).sum(-1).clamp(min=1e-300)
                sites.setdefault((cur["layer"], ch), []).append(pr)
            return comps
        return _w

    with patched_attrs(model, {"_fock_layer_step": _wrap_layer}):
        with patched_attrs(model.V_theta, {"context_components": _wrap_comps}):
            with _eval_mode(model):
                # NOT no_grad: this model family computes its conservative
                # force inside the forward via torch.autograd.grad(V, h),
                # so disabling grad breaks the forward itself. The collected
                # factors are detached individually instead.
                with torch.enable_grad():
                    model(x)

    if not sites:
        return ProbeResult(probe_name="sigma_lr_spectrum_by_site",
                           metrics={"n_sites": 0})

    per_site = {k: torch.cat(v) for k, v in sorted(sites.items())}
    medians = torch.tensor([v.median() for v in per_site.values()],
                           dtype=torch.float64)
    within = torch.tensor([v.std() for v in per_site.values()],
                          dtype=torch.float64)
    between_std, within_std = float(medians.std()), float(within.mean())
    frac = between_std ** 2 / max(between_std ** 2 + within_std ** 2, 1e-300)

    if verbose:
        n_ch = 1 + max(k[1] for k in per_site)
        print(f"PR by (layer, channel) -- {len(per_site)} sites\n")
        print(f"{'layer':>6}" + "".join(f"{f'ch{c}':>9}" for c in range(n_ch)))
        for li in sorted({k[0] for k in per_site}):
            row = "".join(f"{float(per_site[(li, c)].median()):>9.2f}"
                          if (li, c) in per_site else f"{'--':>9}"
                          for c in range(n_ch))
            print(f"{li:>6}{row}")
        print("")
        print(f"site medians: min {float(medians.min()):.2f}  "
              f"max {float(medians.max()):.2f}  "
              f"spread {float(medians.max() - medians.min()):.2f}")
        print(f"between-site std {between_std:.3f} | within-site std {within_std:.3f}"
              f" | between_frac {frac:.2f}")
        print("  -> " + ("STRUCTURED: per-bank rank is the better instrument"
                         if frac > 0.5 else
                         "HOMOGENEOUS: a single global rank is appropriate"))

    return ProbeResult(
        probe_name="sigma_lr_spectrum_by_site",
        metrics={"n_sites": len(per_site),
                 "site_median_min": float(medians.min()),
                 "site_median_max": float(medians.max()),
                 "between_std": between_std, "within_std": within_std,
                 "between_frac": frac},
        raw={"site_medians": {f"L{k[0]}c{k[1]}": float(v.median())
                              for k, v in per_site.items()}})
