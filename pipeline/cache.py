"""Processed-data cache: `outputs/cache/processed/<key>/visits.npz` + `meta.json`.

One float32 matrix of all visit-seconds (meta columns, features, presence flags), per-visit row offsets and ID arrays.
Key = hash of `PrepSettings` + the source of `pipeline/prepare/` and `inventory/` + the per-visit HR sources read
from the inclusion CSV (they decide which signal is the HR feature), so uncommitted edits also
invalidate it. Raw data is treated as immutable; `rebuild=True` forces a rebuild. Validity verdicts are never
baked in: other CSV edits (e.g. the NIRS review) keep the key, and `pipeline.inclusion` joins them at load.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from inventory.build_inclusion import INCLUSION_CSV
from inventory.catalog import RAW_ROOT
from pipeline.prepare.hr import read_hr_sources
from pipeline.prepare.settings import PrepSettings
from pipeline.prepare.visit import ID_FIELDS, META_COLUMNS, ProcessedVisit, prepare_all
from pipeline.provenance import REPO_ROOT, source_hash, utc_now

logger = logging.getLogger(__name__)

CACHE_ROOT = REPO_ROOT / "outputs" / "cache" / "processed"
SOURCE_ROOTS = (REPO_ROOT / "pipeline" / "prepare", REPO_ROOT / "inventory")
CACHE_FORMAT = 1  # bump when the npz layout changes
NPZ_NAME, META_NAME = "visits.npz", "meta.json"
TMP_SUFFIX = ".tmp"


@dataclass(frozen=True)
class ProcessedCache:
    key: str
    columns: tuple[str, ...]
    matrix: np.ndarray  # float32 (n_rows, n_columns), read-only
    visits: pd.DataFrame  # ID_FIELDS + start, stop: the visit's rows are matrix[start:stop]

    @property
    def n_rows(self) -> int:
        return self.matrix.shape[0]

    def column(self, name: str) -> np.ndarray:
        if name not in self.columns:
            raise KeyError(f"column {name!r} not in cache {self.key[:12]} (has {list(self.columns)})")
        return self.matrix[:, self.columns.index(name)]

    def row_visit(self) -> np.ndarray:
        """Visit index (row of `visits`) for every matrix row."""
        lengths = (self.visits["stop"] - self.visits["start"]).to_numpy()
        return np.repeat(np.arange(len(self.visits)), lengths)

    def row_ids(self, field: str) -> np.ndarray:
        """An ID field (e.g. participant_id) for every matrix row."""
        return self.visits[field].to_numpy()[self.row_visit()]


def cache_key(settings: PrepSettings, source_roots: Sequence[Path] = SOURCE_ROOTS,
              hr_sources: Mapping[str, str] | None = None) -> str:
    payload = {"format": CACHE_FORMAT, "settings": asdict(settings), "source": source_hash(source_roots),
               "hr_sources": dict(sorted((hr_sources or {}).items()))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def from_visits(processed: Sequence[ProcessedVisit], key: str) -> ProcessedCache:
    if not processed:
        raise ValueError("no visits to cache")
    columns = tuple(processed[0].columns)
    if columns[: len(META_COLUMNS)] != META_COLUMNS:
        raise ValueError(f"first columns must be {META_COLUMNS}, got {columns[:len(META_COLUMNS)]}")
    blocks, starts, stop = [], [], 0
    for visit in processed:
        if tuple(visit.columns) != columns:
            raise ValueError(f"{visit.ids['visit_id']}: columns {list(visit.columns)} differ from {list(columns)}")
        block = np.column_stack([np.asarray(visit.columns[c], dtype=np.float32) for c in columns])
        blocks.append(block)
        starts.append(stop)
        stop += len(block)
    visits = pd.DataFrame([v.ids for v in processed], columns=list(ID_FIELDS))
    visits = visits.assign(start=starts, stop=[*starts[1:], stop])
    if visits["visit_id"].duplicated().any():
        raise ValueError(f"duplicate visit_ids: {visits.loc[visits['visit_id'].duplicated(), 'visit_id'].tolist()}")
    matrix = np.concatenate(blocks)
    matrix.setflags(write=False)
    return ProcessedCache(key, columns, matrix, visits)


def save(cache: ProcessedCache, folder: Path, settings: PrepSettings,
         hr_sources: Mapping[str, str] | None = None) -> None:
    folder.mkdir(parents=True)
    ids = {field: cache.visits[field].to_numpy(dtype=str) for field in ID_FIELDS}
    offsets = np.append(cache.visits["start"].to_numpy(), cache.n_rows)
    np.savez(folder / NPZ_NAME, matrix=cache.matrix, columns=np.array(cache.columns), offsets=offsets, **ids)
    meta = {"key": cache.key, "format": CACHE_FORMAT, "settings": asdict(settings), "created": utc_now(),
            "n_visits": len(cache.visits), "n_rows": cache.n_rows, "columns": list(cache.columns),
            "hr_sources": dict(sorted((hr_sources or {}).items()))}
    (folder / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load(folder: Path, key: str) -> ProcessedCache:
    with np.load(folder / NPZ_NAME, allow_pickle=False) as npz:
        matrix = npz["matrix"]
        offsets = npz["offsets"]
        visits = pd.DataFrame({field: npz[field].astype(str) for field in ID_FIELDS})
        columns = tuple(str(c) for c in npz["columns"])
    matrix.setflags(write=False)
    visits = visits.assign(start=offsets[:-1], stop=offsets[1:])
    return ProcessedCache(key, columns, matrix, visits)


def load_or_build(settings: PrepSettings = PrepSettings(), rebuild: bool = False, cache_root: Path = CACHE_ROOT,
                  raw_root: Path = RAW_ROOT, inclusion_path: Path = INCLUSION_CSV) -> ProcessedCache:
    hr_sources = read_hr_sources(inclusion_path)
    key = cache_key(settings, hr_sources=hr_sources)
    folder = cache_root / key
    if folder.exists() and not rebuild:
        logger.info("cache hit %s", folder)
        return load(folder, key)
    tmp = folder.with_name(folder.name + TMP_SUFFIX)
    if tmp.exists():
        logger.info("removing leftover %s", tmp)
        shutil.rmtree(tmp)
    logger.info("building cache %s from %s", key[:12], raw_root)
    cache = from_visits(prepare_all(settings, raw_root, hr_sources=hr_sources), key)
    save(cache, tmp, settings, hr_sources)
    if folder.exists():  # rebuild=True: replace only once the new cache is complete
        shutil.rmtree(folder)
    tmp.rename(folder)
    logger.info("cache written: %d visits, %d rows -> %s", len(cache.visits), cache.n_rows, folder)
    return cache
