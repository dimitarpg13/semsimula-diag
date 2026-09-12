"""Is one row of the batch driving the update?

Ported from Cell 6d's ``attribute_spike_rows`` (Diagnostic Programme SS7.3).

The premise (SS39): per-group clipping caps a group at the same ceiling on
every step, quiet or spiking, so a localized event's *applied* update is the
same SIZE either way. Its damage therefore cannot be magnitude -- it has to
be DIRECTION. This replays the captured batch ONE ROW AT A TIME, each scaled
by its own share, and reports how concentrated the gradient is.

**Caveat that travels with this probe** (SS41.5, SS14.4 item 2): on chronic
mechanism-A events the per-row *ranking* survives but the per-row
*magnitudes* are not reliable, because the rows interact through the shared
V_theta geometry rather than contributing independently.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..report import ProbeResult
from ._engine import restored_model_state
from .context import ProbeContext

__all__ = ["attribute_spike_rows"]


def attribute_spike_rows(ctx: ProbeContext, step_tag: int,
                         track: Sequence[str] = ("V_theta.depth_code",),
                         microbatch: Optional[int] = None,
                         top_k: int = 5,
                         verbose: bool = True) -> ProbeResult:
    """Per-row gradient attribution for a captured spike.

    Args:
        track: parameter names to attribute. The default follows the
            notebook; pass the names ``layer_profile`` flagged as dominant.
        microbatch: which microbatch to decompose; ``None`` = all.

    Returns concentration statistics -- what share of the total per-row
    gradient norm sits in the top 1 and top 3 rows.
    """
    ctx.require("store", "forward_fn")
    bundle, path = ctx.store.load(step_tag)
    if verbose:
        print(f'[rows] loaded {path.name}  step={bundle["step"]}')

    model = ctx.model
    grad_accum = ctx.grad_accum or bundle.get("grad_accum",
                                              len(bundle["batches"]))
    mb_idxs = (range(len(bundle["batches"])) if microbatch is None
               else [microbatch])
    tracked = {n: p for n, p in model.named_parameters() if n in set(track)}
    if not tracked:
        raise ValueError(
            f"none of track={list(track)} matched a parameter name; "
            f"pass names as they appear in model.named_parameters()")

    rows: List[Dict[str, Any]] = []
    with restored_model_state(ctx):
        model.load_state_dict(
            {k: v.to(ctx.device) for k, v in bundle["model_state_dict"].items()},
            strict=False)
        model.train()
        for mb in mb_idxs:
            xb, yb = bundle["batches"][mb]
            x_all = torch.as_tensor(xb).long().to(ctx.device)
            y_all = torch.as_tensor(yb).long().to(ctx.device)
            n_rows = x_all.shape[0]
            for row in range(n_rows):
                # RNG is re-pinned per row so every row sees the same stream;
                # otherwise row i's routing depends on how many rows ran first.
                torch.set_rng_state(bundle["rng_state_cpu"])
                if (bundle.get("rng_state_cuda") is not None
                        and ctx.device == "cuda" and torch.cuda.is_available()):
                    torch.cuda.set_rng_state_all(bundle["rng_state_cuda"])
                for p in model.parameters():
                    p.grad = None
                x = x_all[row:row + 1]
                y = y_all[row:row + 1]
                loss, _ntp, _v, _f = ctx.forward_fn(model, x, y, ctx)
                if ctx.register_repulsion and hasattr(model, "pop_repulsion_loss"):
                    loss = loss + model.pop_repulsion_loss()
                # each row's own share of the aggregate
                (loss / (grad_accum * n_rows)).backward()
                entry = {"microbatch": mb, "row": row}
                for name, p in tracked.items():
                    entry[name] = (float(p.grad.detach().norm())
                                   if p.grad is not None else 0.0)
                entry["total"] = sum(v for k, v in entry.items()
                                     if k not in ("microbatch", "row"))
                rows.append(entry)

    rows.sort(key=lambda r: -r["total"])
    grand = sum(r["total"] for r in rows) or 1e-12
    top1 = rows[0]["total"] / grand if rows else 0.0
    top3 = sum(r["total"] for r in rows[:3]) / grand if rows else 0.0

    if verbose:
        print(f'{"mb":>3} {"row":>4} ' +
              " ".join(f"{n.split('.')[-1]:>16}" for n in tracked) +
              f' {"total":>12}')
        for r in rows[:top_k]:
            print(f'{r["microbatch"]:3d} {r["row"]:4d} ' +
                  " ".join(f'{r[n]:16.4f}' for n in tracked) +
                  f' {r["total"]:12.4f}')
        print(f"\n[rows] top-1 share {top1:.1%}, top-3 share {top3:.1%} "
              f"of {len(rows)} rows")
        print("[rows] NOTE (SS41.5): on chronic mechanism-A events the "
              "ranking is reliable but the magnitudes are not.")

    return ProbeResult(
        probe_name="row_attribution",
        step_tag=bundle.get("step", step_tag),
        metrics={"n_rows": len(rows), "top1_share": top1, "top3_share": top3},
        raw={"rows": rows, "tracked": list(tracked)},
    )
