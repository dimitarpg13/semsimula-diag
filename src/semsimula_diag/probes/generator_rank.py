"""Effective rank of V_theta's well-bank generator.

V_theta does not store its Gaussian wells; it *generates* them per token,
per layer, through three dense maps from the concatenated xi context:

    W_mu : xi_dim -> K*d        well centres
    W_a  : xi_dim -> K*d        diagonal precisions
    W_B  : xi_dim -> K*d*rank   low-rank precision factors

At the deployed d=384, K=8, n_ctx=5, rank=4 these carry 35.4M parameters --
93.9% of the model's non-embedding total and 87.6% of its per-token
inference cost, against 24576 MAC of actual *use* per token per layer. The
companion note
``Fock_Inference_Productionization_Plan.md`` §2.3 measures that ratio at
1441:1 and proposes factorising the three maps through a bottleneck.

This module tests that proposal **without retraining**. It reads the
spectrum of each trained map, and optionally truncates in place and
re-evaluates, restoring the weights afterwards.

Read the asymmetry. If perplexity *survives* truncation to rank r, the
directions beyond r were doing nothing: the model had no opportunity to
adapt, so this is strong evidence. If perplexity *collapses*, a trained
factorisation might still recover, so a collapse is weaker evidence against
the proposal than survival is for it. SVD truncation is optimal in
Frobenius norm, not for the task, so it is a lower bound on what a trained
factorisation could reach.

This is a different question from ``ANISO_RANK``. ``probes.stiffness``
measures the rank of each realised ``B_k``, and found a participation ratio
of 3.68 against rank 4 -- all four columns used. That says nothing about
how many of the ``xi_dim`` input directions the generator needs, which has
never been measured.

Weight-only: no gradients, no bundles, no ``forward_fn`` unless an
``evaluate_fn`` is supplied for the truncation sweep.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..report import ProbeResult
from .context import ProbeContext

__all__ = [
    "generator_spectrum",
    "truncation_sweep",
    "GENERATOR_SUFFIXES",
]

#: Note on scope: ``probes.precision_cap._svd_truncate`` also truncates by
#: SVD, but on the *realised* ``B_k`` of shape ``(..., d, rank)`` per token,
#: using a Gram shortcut because a literal per-token SVD is what makes the
#: ``baoab_cfc_lowrank`` arm non-viable. This module truncates a single
#: 2-D weight matrix once, where a plain ``torch.linalg.svd`` is both
#: correct and cheap. The two are not interchangeable.
#:
#: The three maps that constitute the generator. Matched by suffix so the
#: probe is indifferent to how deeply the bank is nested.
GENERATOR_SUFFIXES: Tuple[str, ...] = (
    "B_proj.weight", "mu_proj.weight", "a_proj.weight",
)


def _generator_params(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    v_theta = getattr(model, "V_theta", None)
    if v_theta is None:
        raise ValueError("model has no V_theta; this probe is specific to it")
    found = {n: p for n, p in v_theta.named_parameters()
             if n.endswith(GENERATOR_SUFFIXES)}
    if not found:
        raise ValueError(
            "no B_proj/mu_proj/a_proj found on V_theta. This probe targets "
            "the anisotropic Gaussian bank; an MLP V_theta has no generator "
            "in this sense.")
    return found


def _energy_ranks(s: torch.Tensor,
                  fractions: Sequence[float]) -> Dict[float, int]:
    """Smallest rank capturing each fraction of the squared spectrum.

    Returns 0 for every fraction on an all-zero matrix. That is not a
    pathological input here: ``a_proj`` and ``w_proj`` are zero-initialised
    by ``AnisotropicGaussianVTheta``, so a freshly built model has genuinely
    rank-0 maps and the probe must report that rather than divide by zero.
    """
    e = s.double() ** 2
    total = e.sum()
    if not torch.isfinite(total) or total <= 0:
        return {f: 0 for f in fractions}
    cum = torch.cumsum(e, 0) / total
    return {f: int((cum < f).sum().item()) + 1 for f in fractions}


def generator_spectrum(
    ctx: ProbeContext,
    *,
    fractions: Sequence[float] = (0.90, 0.99, 0.999),
    verbose: bool = True,
) -> ProbeResult:
    """Singular-value spectrum of each generator map.

    Returns participation ratio and the rank needed for each energy
    fraction. Costs one SVD per map and no forward passes.
    """
    params = _generator_params(ctx.model)
    rows, metrics = {}, {}
    if verbose:
        head = f'{"projection":<34}{"shape":>15}{"PR":>9}'
        head += "".join(f'{"r@" + f"{f:g}":>9}' for f in fractions)
        print(head)
    for name, p in sorted(params.items(), key=lambda kv: -kv[1].numel()):
        w = p.detach().float().cpu()
        s = torch.linalg.svdvals(w)
        e = s.double() ** 2
        denom = (e ** 2).sum()
        pr = float((e.sum() ** 2) / denom) if denom > 0 else 0.0
        ranks = _energy_ranks(s, fractions)
        rows[name] = {
            "shape": tuple(w.shape), "participation_ratio": pr,
            "ranks": ranks, "n_params": int(p.numel()),
            "max_rank": int(s.numel()),
        }
        metrics[f"{name}.pr"] = pr
        for f, r in ranks.items():
            metrics[f"{name}.r@{f:g}"] = float(r)
        if verbose:
            line = f'{name:<34}{str(tuple(w.shape)):>15}{pr:>9.1f}'
            line += "".join(f"{ranks[f]:>9d}" for f in fractions)
            print(line)
        del w, s
    total = sum(v["n_params"] for v in rows.values())
    metrics["generator_params"] = float(total)
    if verbose:
        print(f'\n  generator total: {total:,} parameters across '
              f'{len(rows)} maps')
    return ProbeResult(probe_name="generator_spectrum",
                       metrics=metrics, raw=rows)


def truncation_sweep(
    ctx: ProbeContext,
    evaluate_fn: Callable[[], float],
    *,
    ranks: Sequence[int] = (1024, 512, 256, 128, 64, 32),
    verbose: bool = True,
) -> ProbeResult:
    """SVD-truncate the generator in place at each rank and evaluate.

    Args:
        evaluate_fn: zero-argument callable returning a validation loss in
            nats. The notebook's ``evaluate`` satisfies this.
        ranks: truncation ranks, evaluated in the order given.

    The original weights are restored in a ``finally``, so an interrupted
    sweep does not leave the model truncated.
    """
    params = _generator_params(ctx.model)
    original = {n: p.detach().clone() for n, p in params.items()}
    # Cost of a rank-r factorisation of all maps, for reading the saving.
    per_rank_cost = sum(w.shape[0] + w.shape[1] for w in original.values())
    full = sum(int(w.numel()) for w in original.values())
    rows, metrics = [], {}
    try:
        base = float(evaluate_fn())
        base_ppl = float(torch.tensor(base).exp())
        metrics["baseline_loss"] = base
        metrics["baseline_ppl"] = base_ppl
        if verbose:
            print(f'baseline  val_loss {base:.4f}   ppl {base_ppl:.2f}')
            print(f'\n{"rank":>7}{"factorised":>14}{"of full":>10}'
                  f'{"val_loss":>11}{"ppl":>9}{"d ppl":>9}')
        for r in ranks:
            kept = 0
            with torch.no_grad():
                for name, p in params.items():
                    w = original[name].float()
                    u, s, vh = torch.linalg.svd(w, full_matrices=False)
                    k = min(int(r), int(s.numel()))
                    p.copy_(((u[:, :k] * s[:k]) @ vh[:k]).to(p.dtype))
                    kept += k * (w.shape[0] + w.shape[1])
                    del u, s, vh, w
            loss = float(evaluate_fn())
            ppl = float(torch.tensor(loss).exp())
            rows.append({"rank": int(r), "factorised_params": int(kept),
                         "frac_of_full": kept / full,
                         "val_loss": loss, "val_ppl": ppl,
                         "d_ppl": ppl - base_ppl})
            metrics[f"ppl@r{r}"] = ppl
            if verbose:
                print(f'{r:>7d}{kept:>14,}{kept / full:>9.1%}'
                      f'{loss:>11.4f}{ppl:>9.2f}{ppl - base_ppl:>+9.2f}')
    finally:
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(original[name])
        if verbose:
            print('\n  weights restored')
    metrics["break_even_rank"] = float(full / max(per_rank_cost, 1))
    if verbose:
        print(f'  factorising is cheaper than full rank below r = '
              f'{full / per_rank_cost:.0f}')
    return ProbeResult(probe_name="truncation_sweep",
                       metrics=metrics, raw={"rows": rows, "full_params": full})
