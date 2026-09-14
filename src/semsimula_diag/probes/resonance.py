"""Is the explicit low-rank kick running past its stability wall?

Implements D2 and D2b of the resonance hypothesis (semsimula-paper
``companion_notes/Resonance_Hypothesis_for_Gradient_Spikes_in_the_LowRank_Kick.md``).

**The quantity.** Under ``integrator='baoab_cfc'`` the diagonal channel of
$V_\\theta$ is integrated exactly by the closed-form harmonic propagator,
but the off-diagonal part ``L = G G^T`` is demoted to the explicit kick.
Explicit integration of a harmonic mode is stable only while

    omega * dt < 2,      omega = sqrt(lambda_max(L) / m),

so ``omega*dt`` is computable during the FORWARD pass, before a gradient
exists. That is the whole operational point: the live watchdog triggers
post-hoc on gradient norm, whereas this is a leading indicator.

**Why it is affordable, when ``baoab_cfc_lowrank`` is not.** That arm needs
the full eigendecomposition of ``L`` per token, which ``cfc_baoab.py``
measures at ~120 s/step against ``baoab_cfc``'s ~10-15. Here only the TOP
eigenvalue is wanted, and that comes from power iteration using nothing but
matvecs ``L v = G (G^T v)`` -- no eigensolver, no cuSOLVER, GPU-friendly
matmuls only.

**The two hooks.** ``harmonic_terms`` is what the ``baoab_cfc`` path calls,
and it receives exactly the ``(xis, h)`` the layer linearises at; calling
``harmonic_terms_lowrank`` on the same inputs reconstructs the ``G`` the
low-rank arm *would* have built, without changing what the model does.
``cfc_substep`` is patched alongside it purely to read the mass and step the
layer actually used, rather than assuming them.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch

from ..report import ProbeResult
from ._engine import (CURRENT_MICROBATCH, iter_comps, patched_attrs,
                      replayed, restored_model_state, rewrap_comps)
from .precision_cap import _svd_truncate
from .context import ProbeContext

__all__ = ["observe", "omega_dt_report", "omega_dt_under_truncation",
           "tail_coherence_report"]

_SEED = 20260913


def _lambda_max(G: torch.Tensor, n_iter: int, seed: int) -> torch.Tensor:
    """Top eigenvalue of ``L = G G^T`` by power iteration, batched.

    ``G`` is ``(..., d, P)``; the result is ``(...)``. Never forms ``L``
    (which would be ``d x d`` per token) and never calls an eigensolver --
    each iteration is two matmuls against the thin factor.

    **The error is one-sided, in the safe direction for a screen and the
    unsafe one for a clearance.** The returned Rayleigh quotient of a
    not-yet-converged vector is always an UNDER-estimate of the true
    ``lambda_max``, so this can report "under the wall" for a token that is
    actually over it, never the reverse. Convergence is slow exactly when
    the top two eigenvalues are close, which is also when it matters least
    (a near-degenerate pair means no single mode dominates). Treat a
    reading near the wall as "at least this large" and re-measure with more
    iterations before concluding a checkpoint is clear.
    """
    gen = torch.Generator(device=G.device).manual_seed(seed)
    v = torch.randn(G.shape[:-1] + (1,), generator=gen,
                    device=G.device, dtype=G.dtype)
    v = v / v.norm(dim=-2, keepdim=True).clamp(min=1e-30)
    for _ in range(n_iter):
        w = G @ (G.transpose(-2, -1) @ v)
        v = w / w.norm(dim=-2, keepdim=True).clamp(min=1e-30)
    w = G @ (G.transpose(-2, -1) @ v)
    return (v * w).sum(dim=-2).squeeze(-1)        # Rayleigh quotient


_QUANTILE_CAP = 1 << 23          # torch.quantile refuses inputs beyond ~2^24


def _pct(t: torch.Tensor, qs=(0.5, 0.95, 0.99, 1.0)) -> Dict[str, float]:
    flat = t.flatten().float()
    if flat.numel() > _QUANTILE_CAP:
        # max stays exact; the percentiles are estimated on a stride sample
        step = flat.numel() // _QUANTILE_CAP + 1
        sample = flat[::step]
    else:
        sample = flat
    out = {}
    for q in qs:
        key = "max" if q == 1.0 else f"p{int(q * 100)}"
        out[key] = float(flat.max() if q == 1.0
                         else torch.quantile(sample, q))
    return out


class ResonanceMonitor:
    """Accumulates ``omega*dt`` per layer across forward passes."""

    def __init__(self, wall: float = 2.0):
        self.wall = wall
        self.per_layer: Dict[int, List[torch.Tensor]] = {}
        # (microbatch, layer) -> omega*dt. The flat `per_layer` view pools
        # microbatches together, which is fine for a max but hides where a
        # spike sits: at step 87196 the whole event lived in microbatch 2.
        self.by_mb_layer: Dict[tuple, torch.Tensor] = {}
        self._pending: Optional[torch.Tensor] = None
        self._layer = 0
        self._have_layer_idx = False
        self._fallback_layer = 0
        self.dt_substep: Optional[float] = None

    def _stash(self, lam: torch.Tensor) -> None:
        self._pending = lam

    def _finish(self, m: torch.Tensor, dt_substep: float) -> None:
        if self._pending is None:
            return
        self.dt_substep = dt_substep
        # The A substep runs for dt/2, but in `baoab_cfc` the low-rank part
        # sits in the KICK, which runs for the full dt -- and the kick is
        # what carries the stability wall. So the step that matters is
        # twice what cfc_substep was handed.
        dt_kick = 2.0 * dt_substep
        m_b = m if torch.is_tensor(m) else torch.tensor(float(m))
        while m_b.dim() < self._pending.dim() + 1:
            m_b = m_b.unsqueeze(0)
        m_flat = m_b.squeeze(-1) if m_b.shape[-1] == 1 else m_b
        omega = (self._pending / m_flat.clamp(min=1e-30)).clamp(min=0).sqrt()
        vals = (omega * dt_kick).detach().flatten().cpu()
        if self._have_layer_idx:
            layer = self._layer
        else:                       # no _fock_layer_step to read an index from
            layer = self._fallback_layer
            self._fallback_layer += 1
        mb = CURRENT_MICROBATCH.get()
        # setdefault, not assignment: the layer step is re-entered on the
        # gradient-checkpoint recompute, and the FIRST reading is the forward
        # pass. Overwriting would silently report recompute values instead.
        if (mb, layer) not in self.by_mb_layer:
            self.by_mb_layer[(mb, layer)] = vals
            self.per_layer.setdefault(layer, []).append(vals)
        self._pending = None

    def note_layer(self, layer_idx: int) -> None:
        """Record which layer the step about to run belongs to.

        The `_fock_layer_step` hook is the only place the true index is
        available -- neither `harmonic_terms` nor `cfc_substep` receives one.
        The microbatch is NOT inferred here; it comes from the engine's
        `CURRENT_MICROBATCH`, because this hook fires several times per
        (microbatch, layer) under gradient checkpointing and any counting
        rule based on it over-counts.
        """
        self._have_layer_idx = True
        self._layer = layer_idx

    def summary(self) -> Dict[str, Any]:
        """Percentiles of ``omega*dt`` overall and per layer, plus the
        fraction of (token, layer) pairs past the wall."""
        if not self.per_layer:
            return {}
        allv = torch.cat([torch.cat(v) for v in self.per_layer.values()])
        out: Dict[str, Any] = {
            "overall": _pct(allv),
            "frac_over_wall": float((allv > self.wall).float().mean()),
            "wall": self.wall,
            "dt_substep": self.dt_substep,
            "per_layer": {},
        }
        for k, v in sorted(self.per_layer.items()):
            t = torch.cat(v)
            out["per_layer"][k] = {
                **_pct(t), "frac_over_wall": float((t > self.wall).float().mean())}
        out["by_mb_layer"] = {
            k: {"max": float(v.max()), "p50": float(v.median()),
                "frac_over_wall": float((v > self.wall).float().mean())}
            for k, v in sorted(self.by_mb_layer.items())}
        return out


@contextlib.contextmanager
def observe(model, integrator_module, *, wall: float = 2.0,
            n_power_iter: int = 24, seed: int = _SEED
            ) -> Iterator[ResonanceMonitor]:
    """Record ``omega*dt`` for every layer of every forward in the block.

    Read-only: the model's own computation is untouched: the hook calls
    ``harmonic_terms_lowrank`` on the same inputs the layer already
    linearised at, under ``no_grad``, and discards everything but ``G``.

    Args:
        integrator_module: the module the model imported ``cfc_substep``
            from (``model_parf_multixi``, typically). Patched only to read
            the mass and step actually in use.

    Safe to wrap around a training step; the cost is one extra
    ``harmonic_terms_lowrank`` plus ``n_power_iter`` pairs of thin matmuls
    per layer.
    """
    vt = model.V_theta
    if not hasattr(vt, "harmonic_terms_lowrank"):
        raise RuntimeError(
            "V_theta exposes no harmonic_terms_lowrank, so the low-rank "
            "operator L = G G^T cannot be reconstructed. This probe needs "
            "the anisotropic Gaussian family.")
    mon = ResonanceMonitor(wall=wall)

    def _wrap_harmonic(original):
        def _wrapped(xis, h, *, comps=None):
            out = original(xis, h, comps=comps)
            with torch.no_grad():
                _, _, G, _ = vt.harmonic_terms_lowrank(xis, h, comps=comps)
                if G.shape[-1]:
                    mon._stash(_lambda_max(G, n_power_iter, seed))
            return out
        return _wrapped

    def _wrap_substep(original):
        def _wrapped(h, v, f_harm, k_diag, m, dt):
            mon._finish(m, float(dt))
            return original(h, v, f_harm, k_diag, m, dt)
        return _wrapped

    def _wrap_layer(original):
        def _wrapped(h, h_prev, r, salience, m_b, gamma, dt, layer_idx,
                     *a, **kw):
            mon.note_layer(layer_idx)
            return original(h, h_prev, r, salience, m_b, gamma, dt,
                            layer_idx, *a, **kw)
        return _wrapped

    layer_cm = (patched_attrs(model, {"_fock_layer_step": _wrap_layer})
                if hasattr(model, "_fock_layer_step")
                else contextlib.nullcontext())
    with patched_attrs(vt, {"harmonic_terms": _wrap_harmonic}):
        with patched_attrs(integrator_module, {"cfc_substep": _wrap_substep}):
            with layer_cm:
                yield mon


def omega_dt_report(
    ctx: ProbeContext, step_tag: int, integrator_module,
    *, wall: float = 2.0, n_power_iter: int = 24, verbose: bool = True,
) -> ProbeResult:
    """D2: the ``omega*dt`` distribution at one captured checkpoint.

    Replays the capture with the monitor installed and reports how far the
    stiffest mode is from the wall, overall and per layer.

    The hypothesis makes a specific prediction, not merely "something was
    large": inverting the compounding arithmetic against the measured
    897x spike ratio at step 87196 gives ``omega*dt`` of about 2.18, i.e.
    roughly 9% past the wall. A spike capture reading well under 2, or far
    above it, is evidence against the mechanism rather than for it. Run a
    healthy checkpoint alongside for the contrast.
    """
    ctx.require("store", "forward_fn")
    bundle, path = ctx.store.load(step_tag)
    if verbose:
        print(f'[resonance] loaded {path.name}  step={bundle["step"]}')

    with restored_model_state(ctx, grads=False, weights=False, rng=False):
        with observe(ctx.model, integrator_module, wall=wall,
                     n_power_iter=n_power_iter) as mon:
            with replayed(ctx, bundle):
                pass
    s = mon.summary()
    if not s:
        raise RuntimeError(
            "no omega*dt recorded -- the model's layer step never called "
            "V_theta.harmonic_terms, so it is probably not running "
            "integrator='baoab_cfc'.")
    if verbose:
        print(f'  omega*dt  p50={s["overall"]["p50"]:.3f}  '
              f'p95={s["overall"]["p95"]:.3f}  max={s["overall"]["max"]:.3f}'
              f'   over wall({wall}): {100 * s["frac_over_wall"]:.3f}%')
        for k, v in s["per_layer"].items():
            print(f'    layer {k:<2d} p50={v["p50"]:.3f}  max={v["max"]:.3f}  '
                  f'over={100 * v["frac_over_wall"]:.3f}%')
    return ProbeResult(
        probe_name="omega_dt", step_tag=bundle.get("step", step_tag),
        metrics={"omega_dt_p50": s["overall"]["p50"],
                 "omega_dt_p95": s["overall"]["p95"],
                 "omega_dt_max": s["overall"]["max"],
                 "frac_over_wall": s["frac_over_wall"], "wall": wall},
        per_layer={k: v["max"] for k, v in s["per_layer"].items()},
        raw=s)


def tail_coherence_report(
    ctx: ProbeContext, step_tag: int,
    *, verbose: bool = True,
) -> ProbeResult:
    """D2b: are the wells' WEAKEST directions pointing the same way?

    The participation ratio already collected by ``probes.stiffness`` is a
    per-well statistic and is structurally blind to this: every well can be
    perfectly well-conditioned while their weakest directions all align,
    and it is the aligned sum that sets ``lambda_max(L)``. A coherent tail
    contributes as ``n * c^2`` where an incoherent one contributes as
    ``sqrt(n) * c^2``, so a direction beneath notice in any single well can
    dominate the operator.

    For each well the weakest direction in ``h``-space is the last LEFT
    singular vector of ``B_k``, obtained without a ``d x r`` SVD as
    ``B v_min / s_min`` with ``(v_min, s_min)`` from the ``r x r`` Gram
    matrix. Stacking those unit vectors as ``M`` (n_wells x d), the
    reported ``tail_pr`` is the participation ratio of ``M^T M``, computed
    from the eigenvalues of the small ``M M^T`` since the two agree on all
    nonzero eigenvalues. It runs from 1 (every tail identical) to n_wells
    (mutually orthogonal).

    The signature to look for is ``tail_pr`` collapsing at a spike capture
    while per-well participation ratios stay flat.
    """
    ctx.require("store", "forward_fn")
    bundle, path = ctx.store.load(step_tag)
    if verbose:
        print(f'[tailcoh] loaded {path.name}  step={bundle["step"]}')

    prs: List[torch.Tensor] = []
    cos: List[torch.Tensor] = []

    def _wrap(original):
        def _wrapped(xis):
            comps = original(xis)
            with torch.no_grad():
                for (_mu, _a, _w, B) in iter_comps(comps):
                    if B.shape[-1] < 2:
                        continue
                    gram = (B.transpose(-2, -1) @ B).cpu()
                    evals, evecs = torch.linalg.eigh(gram)
                    v_min = evecs[..., :1]                    # (..., r, 1)
                    s_min = evals[..., :1].clamp(min=1e-12).sqrt()
                    u = (B.cpu() @ v_min).squeeze(-1) / s_min  # (..., K, d)
                    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-30)
                    g = u @ u.transpose(-2, -1)               # (..., K, K)
                    lam = torch.linalg.eigvalsh(g).clamp(min=0)
                    prs.append((lam.sum(-1) ** 2
                                / lam.pow(2).sum(-1).clamp(min=1e-30)).flatten())
                    k = g.shape[-1]
                    off = g.abs().sum((-2, -1)) - g.diagonal(
                        dim1=-2, dim2=-1).abs().sum(-1)
                    cos.append((off / max(k * (k - 1), 1)).flatten())
            return comps
        return _wrapped

    with restored_model_state(ctx, grads=False, weights=False, rng=False):
        with patched_attrs(ctx.model.V_theta, {"context_components": _wrap}):
            with replayed(ctx, bundle):
                pass
    if not prs:
        raise RuntimeError("no well banks with rank >= 2 were observed.")

    pr, cs = torch.cat(prs), torch.cat(cos)
    n_wells = None
    m = {"tail_pr_p50": float(pr.median()),
         "tail_pr_p05": float(torch.quantile(pr, 0.05)),
         "tail_mean_abs_cos_p50": float(cs.median()),
         "tail_mean_abs_cos_p95": float(torch.quantile(cs, 0.95))}
    if verbose:
        print(f'  tail_pr        p05={m["tail_pr_p05"]:.3f}  '
              f'p50={m["tail_pr_p50"]:.3f}   (1 = fully coherent)')
        print(f'  |cos| of tails p50={m["tail_mean_abs_cos_p50"]:.4f}  '
              f'p95={m["tail_mean_abs_cos_p95"]:.4f}')
    return ProbeResult(probe_name="tail_coherence",
                       step_tag=bundle.get("step", step_tag), metrics=m)


def omega_dt_under_truncation(
    ctx: ProbeContext, step_tag: int, integrator_module,
    ranks: Sequence[int] = (3,),
    *, wall: float = 2.0, n_power_iter: int = 24, verbose: bool = True,
) -> Dict[str, ProbeResult]:
    """The decisive within-bundle test: does the truncation that kills the
    spike also carry ``omega*dt`` back under the wall?

    Every spikebatch bundle is a spike capture by construction -- the
    watchdog is what triggers the save -- so there is no "healthy" bundle to
    contrast against. This supplies the contrast from inside one capture
    instead, and it is a sharper test than a cross-checkpoint comparison
    because nothing varies but the truncation.

    The mechanism predicts a specific pairing, and both halves have to hold:
    the untruncated arm reads above ``wall`` and the truncated arm reads
    below it, in the same bundle, at the same tokens and layers. A
    truncation that annihilates the gradient while leaving ``omega*dt``
    unchanged would falsify the account outright -- the gradient would have
    collapsed for some reason having nothing to do with the stability wall.
    """
    ctx.require("store", "forward_fn")
    bundle, path = ctx.store.load(step_tag)
    if verbose:
        print(f'[omega/trunc] loaded {path.name}  step={bundle["step"]}')

    def _run(rank: Optional[int]) -> Dict[str, Any]:
        if rank is None:
            cm: contextlib.AbstractContextManager = contextlib.nullcontext()
        else:
            def _make(original):
                def _wrapped(xis):
                    comps = original(xis)
                    out_c = [(mu, a, w,
                             _svd_truncate(B, rank) if B.shape[-1] else B)
                            for (mu, a, w, B) in iter_comps(comps)]
                    return rewrap_comps(comps, out_c)
                return _wrapped
            cm = patched_attrs(ctx.model.V_theta,
                               {"context_components": _make})
        with cm:
            with observe(ctx.model, integrator_module, wall=wall,
                         n_power_iter=n_power_iter) as mon:
                with replayed(ctx, bundle):
                    pass
        return mon.summary()

    out: Dict[str, ProbeResult] = {}
    with restored_model_state(ctx, grads=False, weights=False, rng=False):
        for label, rank in [("untruncated", None)] + [(f"rank={r}", r)
                                                      for r in ranks]:
            s = _run(rank)
            if not s:
                raise RuntimeError(
                    "no omega*dt recorded -- V_theta.harmonic_terms was "
                    "never called, so the model is probably not running "
                    "integrator='baoab_cfc'.")
            out[label] = ProbeResult(
                probe_name="omega_dt_truncation",
                step_tag=bundle.get("step", step_tag),
                metrics={"omega_dt_p50": s["overall"]["p50"],
                         "omega_dt_p95": s["overall"]["p95"],
                         "omega_dt_max": s["overall"]["max"],
                         "frac_over_wall": s["frac_over_wall"], "wall": wall},
                per_layer={k: v["max"] for k, v in s["per_layer"].items()})
            if verbose:
                print(f'  {label:14s} omega*dt p50={s["overall"]["p50"]:.3f}  '
                      f'max={s["overall"]["max"]:.3f}  '
                      f'over wall={100 * s["frac_over_wall"]:.3f}%')
    return out
