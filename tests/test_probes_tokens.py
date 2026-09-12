"""Token-degeneracy probes. Pure CPU bookkeeping -- no model needed."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from semsimula_diag import BundleStore, ProbeContext
from semsimula_diag.probes import tokens


def test_row_degeneracy_counts_longest_backtoback_run():
    d = tokens.row_degeneracy([1, 1, 1, 2, 3, 3, 4])
    assert d["max_repeat_run"] == 3
    assert d["seq_len"] == 7
    assert d["unique_token_ratio"] == pytest.approx(4 / 7)


def test_row_degeneracy_handles_empty_row():
    assert tokens.row_degeneracy([])["seq_len"] == 0


def test_max_repeat_run_is_blind_to_phrase_level_repetition():
    """SS49.7: the metric only sees back-to-back identical tokens, which is
    why unique_token_ratio caught a templated row it ranked 14/32."""
    templated = [1, 2, 3] * 8          # heavily repetitive, never back-to-back
    d = tokens.row_degeneracy(templated)
    assert d["max_repeat_run"] == 1, "phrase repetition is invisible here"
    assert d["unique_token_ratio"] < 0.2, "but the ratio does see it"


@pytest.fixture
def store(tmp_path):
    """A synthetic bundle: row 0 degenerate, row 1 clean, row 2 templated."""
    # Chosen so BOTH metrics rank the three rows distinctly, with no ties
    # to break -- otherwise the rank assertions below depend on sort
    # stability rather than on the statistic.
    rows = np.array([
        [7] * 16,                       # repeat 16, uniq 0.0625 -- worst on both
        list(range(16)),                # repeat  1, uniq 1.0    -- best on both
        [1, 1, 2, 3] * 4,               # repeat  2, uniq 0.1875 -- middle on both
    ], dtype=np.int64)
    bundle = {
        "step": 4242,
        "batches": [(rows, rows)],
        "grad_accum": 1,
        "pre_clip_grad_norm": 123.4,
        "top_groups": {},
    }
    ckpt = tmp_path / "ckpts"
    ckpt.mkdir()
    torch.save(bundle, ckpt / "run_step4242_spikebatch.pt")
    return BundleStore(ckpt_dir=ckpt, ckpt_prefix="run", archive_root=tmp_path,
                       verbose=False)


@pytest.fixture
def ctx(store):
    return ProbeContext(model=nn.Linear(2, 2), store=store)


def test_inspect_ranks_the_degenerate_row_first(ctx):
    r = tokens.inspect_spike_tokens(ctx, 4242, verbose=False)
    assert r.step_tag == 4242
    assert r.metrics["n_rows"] == 3
    assert r.metrics["max_repeat_run_worst"] == 16
    assert r.raw["rows"][0]["row"] == 0


def test_inspect_never_returns_raw_token_ids(ctx):
    """`ids` are large and not a diagnostic; they stay out of the result."""
    r = tokens.inspect_spike_tokens(ctx, 4242, verbose=False)
    assert all("ids" not in row for row in r.raw["rows"])


def test_decode_hot_rows_ranks_named_rows_against_their_batch(ctx):
    """The SS49.7 direction: attribution names the row, we report its rank."""
    r = tokens.decode_hot_rows(ctx, 4242, hot_rows=[(0, 1)], verbose=False)
    hot = r.raw["hot_rows"][0]
    assert hot["row"] == 1
    assert hot["n_rows"] == 3
    # the clean row ranks LAST by descending repeat-run and LAST by
    # ascending uniqueness -- demonstrably non-degenerate on both metrics,
    # which is exactly how SS49.7 falsified degeneracy as a necessary
    # condition for a row to be implicated by gradient attribution
    assert hot["rank_by_max_repeat_run"] == 3
    assert hot["rank_by_unique_ratio"] == 3


def test_decode_hot_rows_skips_a_row_that_does_not_exist(ctx):
    r = tokens.decode_hot_rows(ctx, 4242, hot_rows=[(0, 99)], verbose=False)
    assert r.metrics["n_hot_rows"] == 0


def test_tokenizer_is_optional(ctx):
    tokens.inspect_spike_tokens(ctx, 4242, tokenizer=None, verbose=True)
