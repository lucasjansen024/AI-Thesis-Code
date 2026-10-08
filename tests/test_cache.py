"""Cache key invalidation and the npz round trip."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.cache import cache_key, load, save
from pipeline.prepare.settings import PrepSettings
from tests.synthetic import Synthetic


@pytest.fixture
def source_roots(tmp_path: Path) -> tuple[Path, Path]:
    prepare, inventory = tmp_path / "prepare", tmp_path / "inventory"
    prepare.mkdir()
    inventory.mkdir()
    (prepare / "grid.py").write_text("X = 1\n", encoding="utf-8")
    (inventory / "sensors.py").write_text("Y = 2\n", encoding="utf-8")
    return prepare, inventory


def test_key_is_stable(source_roots: tuple[Path, Path]) -> None:
    assert cache_key(PrepSettings(), source_roots) == cache_key(PrepSettings(), source_roots)


def test_key_changes_with_a_prep_setting(source_roots: tuple[Path, Path]) -> None:
    default = PrepSettings()
    changed = PrepSettings(max_interp_gap_s=default.max_interp_gap_s + 1)
    assert cache_key(default, source_roots) != cache_key(changed, source_roots)


@pytest.mark.parametrize("which", [0, 1], ids=["prepare", "inventory"])
def test_key_changes_with_source(source_roots: tuple[Path, Path], which: int) -> None:
    before = cache_key(PrepSettings(), source_roots)
    (source_roots[which] / "edited.py").write_text("Z = 3\n", encoding="utf-8")
    after_new_file = cache_key(PrepSettings(), source_roots)
    (source_roots[which] / "edited.py").write_text("Z = 4\n", encoding="utf-8")
    after_edit = cache_key(PrepSettings(), source_roots)
    assert len({before, after_new_file, after_edit}) == 3


def test_key_ignores_bytecode(source_roots: tuple[Path, Path]) -> None:
    before = cache_key(PrepSettings(), source_roots)
    pycache = source_roots[0] / "__pycache__"
    pycache.mkdir()
    (pycache / "grid.cpython-312.py").write_text("junk\n", encoding="utf-8")
    assert cache_key(PrepSettings(), source_roots) == before


def test_default_key_covers_the_real_source() -> None:
    assert len(cache_key(PrepSettings())) == 64


def test_npz_round_trip(synthetic: Synthetic, tmp_path: Path) -> None:
    folder = tmp_path / "key"
    save(synthetic.cache, folder, PrepSettings())
    loaded = load(folder, synthetic.cache.key)
    assert loaded.columns == synthetic.cache.columns
    np.testing.assert_array_equal(loaded.matrix, synthetic.cache.matrix)
    pd.testing.assert_frame_equal(loaded.visits, synthetic.cache.visits, check_dtype=False)
    assert not loaded.matrix.flags.writeable
