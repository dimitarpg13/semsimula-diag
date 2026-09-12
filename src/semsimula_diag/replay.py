"""Deterministic re-run engine: gradient isolation and capture-bundle lookup.

Ported from Cell 6d of
``colab_fock_cfc_baoab_aniso_gaussian_openwebtext_d384.ipynb``
(semsimula-paper repo): ``_isolated_grad_snapshot`` / ``_isolated_grad_restore``
and ``_resolve_bundle_path`` / ``_load_spike_bundle``.

Corresponds to ``semsimula_diag.replay`` in
``companion_notes/Diagnostic_Programme_in_CfC_BAOAB_Integrator.md`` SS11.2-SS11.3
(migration step 2). Model-agnostic by construction: every function here takes
the model or the store explicitly and knows nothing about Fock internals.

**The behavioural change from the notebook**: ``_resolve_bundle_path`` read
``CKPT_PREFIX`` / ``CKPT_DIR`` / ``GDRIVE_ROOT`` out of ``globals()``. Those are
now fields on :class:`BundleStore`. Resolution order is unchanged -- live
checkpoint dir first, then the permanent archive -- as is the decision to
raise a :class:`FileNotFoundError` naming *both* places it looked, so a bundle
evicted by ring rotation is diagnosable rather than merely missing.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "grad_snapshot",
    "grad_restore",
    "isolated_grads",
    "BundleStore",
]


# ---------------------------------------------------------------------------
# Gradient isolation -- the non-pollution invariant every probe depends on
# ---------------------------------------------------------------------------

def grad_snapshot(mdl: nn.Module) -> Dict[str, Optional[torch.Tensor]]:
    """Save existing ``.grad`` tensors (or ``None``) for every parameter."""
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in mdl.named_parameters()}


def grad_restore(
    mdl: nn.Module, saved: Dict[str, Optional[torch.Tensor]]
) -> None:
    """Restore a snapshot taken by :func:`grad_snapshot`."""
    for n, p in mdl.named_parameters():
        g = saved.get(n)
        p.grad = g.clone() if g is not None else None


@contextlib.contextmanager
def isolated_grads(mdl: nn.Module) -> Iterator[None]:
    """Run a probe without disturbing whatever gradients are already on the
    model -- the invariant that lets a diagnostic be called mid-training
    without corrupting the step in flight.

    Restores on the way out even if the body raises, which the notebook's
    bare snapshot/restore pair did not guarantee.
    """
    saved = grad_snapshot(mdl)
    try:
        yield
    finally:
        grad_restore(mdl, saved)


# ---------------------------------------------------------------------------
# Capture-bundle lookup (live ring, then permanent archive)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BundleStore:
    """Where capture bundles live, and how they are named.

    The live rings rotate (``SPIKEBATCH_SNAPSHOT_MAX_KEEP=12``,
    ``PRERELOAD_SNAPSHOT_MAX_KEEP=5`` in the reference run) on every new
    capture, independent of whether anyone has analysed an older bundle yet --
    one 24h session has already produced 13 captures against a 12-deep ring.
    So a bundle worth analysing is frequently *not* in ``ckpt_dir`` any more,
    even though a byte-identical copy was written to the permanent archive.

    Attributes:
        ckpt_dir: the live checkpoint directory the training loop writes to.
        ckpt_prefix: run-identifying filename prefix, i.e. bundles are named
            ``f'{ckpt_prefix}_step{tag}_{suffix}.pt'``.
        archive_root: parent of the ``{suffix}_archive`` folders. Defaults to
            ``ckpt_dir.parent``, which matches a local run; on Colab pass the
            Drive root explicitly (the notebook's ``GDRIVE_ROOT``).
        verbose: print a one-line notice when a bundle is served from the
            archive rather than the live ring, matching the notebook.
    """

    ckpt_dir: Path
    ckpt_prefix: str
    archive_root: Optional[Path] = None
    verbose: bool = True

    def _archive_dir(self, suffix: str) -> Path:
        root = self.archive_root if self.archive_root is not None else self.ckpt_dir.parent
        return root / f'{suffix}_archive'

    def filename(self, step_tag: int, suffix: str = 'spikebatch') -> str:
        return f'{self.ckpt_prefix}_step{step_tag}_{suffix}.pt'

    def resolve(
        self, step_tag: int, suffix: str = 'spikebatch'
    ) -> Tuple[Optional[Path], Optional[str]]:
        """Find one bundle, live dir first then archive.

        Returns ``(path, 'live'|'archive')``, or ``(None, None)`` if neither
        has it.
        """
        fname = self.filename(step_tag, suffix)
        live = self.ckpt_dir / fname
        if live.exists():
            return live, 'live'
        archived = self._archive_dir(suffix) / fname
        if archived.exists():
            return archived, 'archive'
        return None, None

    def load(
        self, step_tag: int, suffix: str = 'spikebatch'
    ) -> Tuple[Any, Path]:
        """Resolve and load one capture bundle. Returns ``(bundle, path)``.

        ``map_location='cpu'`` (not the training device) is load-bearing:
        ``rng_state_cpu`` / ``rng_state_cuda`` must stay plain CPU
        ``ByteTensor`` s for ``torch.(cuda.)set_rng_state`` -- moving them to
        the device here breaks that with "RNG state must be a
        torch.ByteTensor". The ``model_state_dict`` tensors don't need it
        either: callers re-map them per-tensor with ``.to(device)``.

        Raises:
            FileNotFoundError: naming both directories searched, rather than
                letting ``torch.load`` raise a bare one.
        """
        path, src = self.resolve(step_tag, suffix)
        if path is None:
            raise FileNotFoundError(
                f'no {suffix} bundle for step {step_tag}: not in ckpt_dir '
                f'({self.ckpt_dir}) nor in {self._archive_dir(suffix)}. The '
                f'live ring rotates, so it may have been evicted before it '
                f'was ever archived.'
            )
        if src == 'archive' and self.verbose:
            print(f'  [archive] step {step_tag} {suffix} -- evicted from the '
                  f'live ring, loaded from {suffix}_archive')
        return torch.load(path, map_location='cpu', weights_only=False), path
