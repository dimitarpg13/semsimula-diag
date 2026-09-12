"""Shared fixtures, including real capture bundles when they are available.

The ``*_spikebatch.pt`` bundles are ~294 MB each and must never be committed;
they are located at run time instead. Set ``SEMSIMULA_BUNDLES`` to the
directory holding them (defaults to ``~/Downloads``). Tests that need one skip
cleanly when it is absent, so the suite stays green on a machine without them.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

CKPT_PREFIX = (
    'fock_cfc_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_L8probe_ob_untied'
    '_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc'
)

BUNDLE_DIR = Path(os.environ.get('SEMSIMULA_BUNDLES', Path.home() / 'Downloads'))


def bundle_path(step_tag: int, suffix: str = 'spikebatch') -> Path:
    return BUNDLE_DIR / f'{CKPT_PREFIX}_step{step_tag}_{suffix}.pt'


def have_bundle(step_tag: int, suffix: str = 'spikebatch') -> bool:
    return bundle_path(step_tag, suffix).exists()


requires_bundles = pytest.mark.skipif(
    not (have_bundle(87196) and have_bundle(90360)),
    reason=(
        f'capture bundles not found in {BUNDLE_DIR} '
        f'(set SEMSIMULA_BUNDLES to their directory)'
    ),
)


@pytest.fixture(scope='session')
def ckpt_prefix() -> str:
    return CKPT_PREFIX


@pytest.fixture(scope='session')
def bundle_dir() -> Path:
    return BUNDLE_DIR
