"""Hashes, git state, package versions and timestamps for cache keys and manifests."""
from __future__ import annotations

import hashlib
import logging
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "numpy", "pandas", "pyarrow", "pingouin", "scipy", "statsmodels")
SOURCE_SUFFIX = ".py"
SEED_BYTES = 8
SEED_MASK = 2**63 - 1


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash(roots: Iterable[Path]) -> str:
    """Hash of every .py file under the roots: relative path + bytes, in sorted order (caches are skipped)."""
    digest = hashlib.sha256()
    for root in roots:
        files = sorted(p for p in root.rglob(f"*{SOURCE_SUFFIX}") if "__pycache__" not in p.parts)
        if not files:
            raise FileNotFoundError(f"no {SOURCE_SUFFIX} files under {root}")
        for path in files:
            digest.update(path.relative_to(root.parent).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def derive_seed(*parts: object) -> int:
    """A stable 63-bit seed from its parts, e.g. ('init', seed, config_id). Independent of PYTHONHASHSEED."""
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:SEED_BYTES], "big") & SEED_MASK


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("git %s failed (%s); manifest records null", " ".join(args), exc)
        return None
    return result.stdout.strip()


def git_state() -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def package_versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
