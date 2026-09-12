"""What the model was actually reading: token-degeneracy forensics.

Ported from Cell 6d's ``inspect_spike_tokens`` and Cell 6d-3's
``decode_hot_rows`` (Diagnostic Programme SS7.2, SS11.2).

Pure CPU bookkeeping -- no model, no GPU, no RNG or weight state touched --
so these are safe to call at any time, interleaved with training or other
replays. Only a bundle store and (for decoding snippets) a tokenizer are
needed.

The two probes ask different questions and SS49.7 is the reason both exist:

- :func:`inspect_spike_tokens` ranks a capture's rows by degeneracy and
  decodes the worst, *searching* for a degenerate row.
- :func:`decode_hot_rows` takes rows already named by gradient attribution
  and asks where those rank against their own batch -- which is the honest
  direction, and is what showed that degeneracy is **not** a necessary
  condition for a row to be implicated.

``max_repeat_run`` only counts back-to-back identical tokens and is
structurally blind to phrase-level repetition; ``unique_token_ratio`` caught
a heavily templated row that ``max_repeat_run`` ranked 14/32 (SS49.7). Read
both.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..report import ProbeResult
from .context import ProbeContext

__all__ = ["row_degeneracy", "inspect_spike_tokens", "decode_hot_rows"]


def row_degeneracy(ids: Sequence[int]) -> Dict[str, float]:
    """Degeneracy statistics for one row of token ids.

    Returns ``unique_token_ratio`` and ``max_repeat_run`` (longest run of the
    same id back-to-back). Both are needed: see the module docstring.
    """
    n = len(ids)
    if n == 0:
        return {"seq_len": 0, "unique_token_ratio": 0.0, "max_repeat_run": 0}
    best_run = cur_run = 1
    cur_id = ids[0]
    for t in ids[1:]:
        if t == cur_id:
            cur_run += 1
        else:
            cur_run, cur_id = 1, t
        best_run = max(best_run, cur_run)
    return {
        "seq_len": n,
        "unique_token_ratio": len(set(ids)) / n,
        "max_repeat_run": best_run,
    }


def _rows_of(bundle, microbatches: Optional[Iterable[int]]) -> List[Dict[str, Any]]:
    idxs = (range(len(bundle["batches"])) if microbatches is None
            else list(microbatches))
    rows: List[Dict[str, Any]] = []
    for mb in idxs:
        xb, _yb = bundle["batches"][mb]
        for row in range(xb.shape[0]):
            ids = xb[row].tolist()          # numpy array in the bundle
            rows.append({"microbatch": mb, "row": row,
                         **row_degeneracy(ids), "ids": ids})
    return rows


def inspect_spike_tokens(
    ctx: ProbeContext,
    step_tag: int,
    microbatches: Optional[Iterable[int]] = None,
    tokenizer: Any = None,
    show_n: int = 3,
    snippet_chars: int = 240,
    verbose: bool = True,
) -> ProbeResult:
    """Rank a capture's rows by degeneracy and decode the worst offenders.

    Args:
        microbatches: which microbatch indices to inspect; ``None`` = all.
        tokenizer: anything with ``.decode(ids)``. Omit to skip snippets --
            the statistics do not need it.
    """
    ctx.require("store")
    bundle, _ = ctx.store.load(step_tag)
    rows = _rows_of(bundle, microbatches)
    rows.sort(key=lambda r: r["max_repeat_run"], reverse=True)

    if verbose:
        print(f'[tokens] step {bundle["step"]}: {len(rows)} row(s), '
              f'pre_clip_grad_norm={bundle.get("pre_clip_grad_norm")}')
        print(f'{"mb":>3} {"row":>4} {"len":>5} {"uniq_ratio":>10} '
              f'{"max_repeat_run":>14}')
        for r in rows:
            print(f'{r["microbatch"]:3d} {r["row"]:4d} {r["seq_len"]:5d} '
                  f'{r["unique_token_ratio"]:10.3f} {r["max_repeat_run"]:14d}')
        if tokenizer is not None:
            print(f'\n[tokens] decoded snippet for the top '
                  f'{min(show_n, len(rows))} row(s) by max_repeat_run:')
            for r in rows[:show_n]:
                snippet = tokenizer.decode(r["ids"])[:snippet_chars]
                print(f'\n  -- microbatch {r["microbatch"]}, row {r["row"]} '
                      f'(len={r["seq_len"]}, '
                      f'uniq={r["unique_token_ratio"]:.3f}, '
                      f'max_repeat_run={r["max_repeat_run"]}) --')
                print("  " + snippet.replace("\n", "\\n"))

    worst = rows[0] if rows else {}
    return ProbeResult(
        probe_name="tokens",
        step_tag=bundle.get("step", step_tag),
        metrics={
            "n_rows": len(rows),
            "max_repeat_run_worst": float(worst.get("max_repeat_run", 0)),
            "unique_token_ratio_worst": float(worst.get("unique_token_ratio", 0.0)),
        },
        raw={"rows": [{k: v for k, v in r.items() if k != "ids"} for r in rows]},
    )


def decode_hot_rows(
    ctx: ProbeContext,
    step_tag: int,
    hot_rows: Sequence[tuple],
    tokenizer: Any = None,
    snippet_chars: int = 240,
    verbose: bool = True,
) -> ProbeResult:
    """Rank rows *named by gradient attribution* against their own batch.

    This is the direction that matters (SS49.7): rather than searching for a
    degenerate row and hoping it is the culprit, take the rows
    :func:`~semsimula_diag.probes.row_attribution.attribute_spike_rows`
    already implicated and ask where they sit in their batch's own
    distribution. That is how degeneracy was falsified as a *necessary*
    condition -- two of three implicated rows sat on the NON-degenerate side
    by both metrics.

    Args:
        hot_rows: ``[(microbatch, row), ...]`` as named by attribution.
    """
    ctx.require("store")
    bundle, _ = ctx.store.load(step_tag)
    all_rows = _rows_of(bundle, None)

    by_repeat = sorted(all_rows, key=lambda r: -r["max_repeat_run"])
    by_uniq = sorted(all_rows, key=lambda r: r["unique_token_ratio"])
    rank_repeat = {(r["microbatch"], r["row"]): i + 1
                   for i, r in enumerate(by_repeat)}
    rank_uniq = {(r["microbatch"], r["row"]): i + 1
                 for i, r in enumerate(by_uniq)}
    n = len(all_rows)

    out: List[Dict[str, Any]] = []
    for mb, row in hot_rows:
        match = next((r for r in all_rows
                      if r["microbatch"] == mb and r["row"] == row), None)
        if match is None:
            print(f"  [skip] no row {row} in microbatch {mb}")
            continue
        key = (mb, row)
        entry = {k: v for k, v in match.items() if k != "ids"}
        entry["rank_by_max_repeat_run"] = rank_repeat[key]
        entry["rank_by_unique_ratio"] = rank_uniq[key]
        entry["n_rows"] = n
        out.append(entry)
        if verbose:
            print(f'  microbatch {mb} row {row}: '
                  f'max_repeat_run={match["max_repeat_run"]} '
                  f'(rank {rank_repeat[key]}/{n}), '
                  f'unique_ratio={match["unique_token_ratio"]:.3f} '
                  f'(rank {rank_uniq[key]}/{n})')
            if tokenizer is not None:
                snippet = tokenizer.decode(match["ids"])[:snippet_chars]
                print("    " + snippet.replace("\n", "\\n"))

    return ProbeResult(
        probe_name="decode_hot_rows",
        step_tag=bundle.get("step", step_tag),
        metrics={"n_hot_rows": len(out), "n_rows": n},
        raw={"hot_rows": out},
    )
