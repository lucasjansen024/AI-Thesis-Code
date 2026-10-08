"""Markers, skips and shared fixtures.

- `@pytest.mark.real`: needs `data/raw/` (both cohorts); builds or reuses the real processed cache.
- `@pytest.mark.xpu`: needs an Intel XPU device (determinism tests run on the training device only).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from pipeline.cache import ProcessedCache, load_or_build
from pipeline.inclusion import read_inclusion
from pipeline.prepare.settings import PrepSettings
from tests.synthetic import Synthetic, make_synthetic

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_COHORT_DIRS = (REPO_ROOT / "data" / "raw" / "Gies", REPO_ROOT / "data" / "raw" / "Lucas")


def _has_raw() -> bool:
    return all(d.is_dir() for d in RAW_COHORT_DIRS)


def _has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "real: needs data/raw/ for both cohorts (builds the real processed cache)")
    config.addinivalue_line("markers", "xpu: needs an Intel XPU device")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_real = pytest.mark.skip(reason=f"raw data absent: {', '.join(str(d) for d in RAW_COHORT_DIRS)}")
    skip_xpu = pytest.mark.skip(reason="no XPU device (torch.xpu.is_available() is False)")
    has_raw, has_xpu = _has_raw(), _has_xpu()
    for item in items:
        if "real" in item.keywords and not has_raw:
            item.add_marker(skip_real)
        if "xpu" in item.keywords and not has_xpu:
            item.add_marker(skip_xpu)


@pytest.fixture
def synthetic() -> Synthetic:
    return make_synthetic()


@pytest.fixture(scope="session")
def real_cache() -> ProcessedCache:
    """The real processed cache for the default settings (first build ~3 s per visit, then reused)."""
    return load_or_build(PrepSettings())


@pytest.fixture(scope="session")
def committed_inclusion() -> pd.DataFrame:
    return read_inclusion()
