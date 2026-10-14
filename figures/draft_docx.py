"""Shareable Word draft for co-authors: introduction, methods and results from `docs/paper/`, then the three figures
(Tables 1 and 2 and the legends of Figures 2 and 3 are inline in the text), then the reference list. One `.docx` with
the figures embedded.

Needs pandoc: on PATH, or bundled with the `pypandoc_binary` package (`pip install pypandoc_binary`). The generated
display-items markdown is kept next to the `.docx` so the input of every export can be inspected.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import date
from pathlib import Path

from figures.style import PAPER_ROOT
from pipeline.provenance import REPO_ROOT

logger = logging.getLogger(__name__)

PAPER_DOCS = REPO_ROOT / "docs" / "paper"
TEXT_FILES = ("01_introduction.md", "02_methods.md", "03_results.md")
REFERENCES_FILE = "references.md"
SHARE_ROOT = PAPER_ROOT / "share"
DISPLAY_ITEMS_NAME = "display_items.md"
DOCUMENT_TITLE = "Continuous V̇O₂ estimation: draft for co-authors"
DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH = "16cm", "9cm"
FIGURES = (  # (caption, png name in PAPER_ROOT, width); Figure 3's png is passed in, see write_draft_docx
    ("**Figure 1.** Study design. (A) Work-rate profiles of the seven protocols for one participant per cohort, on a "
     "relative intensity scale (GET = 1, peak power = 2). (B) Training and test data of each analysis: Aim 1 "
     "leave-one-subject-out and the Aim 2 left-out conditions and cohort transfers.", "study_design.png",
     DOUBLE_COLUMN_WIDTH),
    ("**Figure 2.** Legend in Results §3.1.", "aim1_dot_plot.png", SINGLE_COLUMN_WIDTH),
)
AIM2_CAPTION = "**Figure 3.** Legend in Results §3.5."
INTERIM_NOTE = " **Interim: 2b Step is still training and not shown.**"


class PandocNotFoundError(RuntimeError):
    """Neither a `pandoc` on PATH nor `pypandoc_binary` is available."""


def pandoc_path() -> str:
    """`pandoc` on PATH, else the binary bundled with `pypandoc_binary`."""
    on_path = shutil.which("pandoc")
    if on_path:
        return on_path
    try:
        import pypandoc
        return pypandoc.get_pandoc_path()
    except (ImportError, OSError) as exc:
        raise PandocNotFoundError("pandoc not found: install pandoc, or run `pip install pypandoc_binary`") from exc


def display_items_markdown(aim2_figure: Path, is_interim: bool) -> str:
    """Figures 1-3 as markdown, images by absolute path."""
    figures = [(caption, PAPER_ROOT / name, width) for caption, name, width in FIGURES]
    figures.append((AIM2_CAPTION + (INTERIM_NOTE if is_interim else ""), aim2_figure, SINGLE_COLUMN_WIDTH))
    blocks = ["# Figures", "Table 1 is in Methods §2.1 and Table 2 in Results §3.1."]
    blocks += [f"![{caption}]({path.as_posix()}){{width={width}}}" for caption, path, width in figures]
    return "\n\n".join(blocks) + "\n"


def write_draft_docx(aim2_figure: Path = PAPER_ROOT / "aim2_generalisation.png", is_interim: bool = False,
                     out_dir: Path = SHARE_ROOT) -> Path:
    """Draft → `<out_dir>/VO2_draft_<date>.docx`; returns its path."""
    inputs = [PAPER_DOCS / name for name in TEXT_FILES] + [aim2_figure]
    inputs += [PAPER_ROOT / name for _, name, _ in FIGURES]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"draft export: missing inputs {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    display_items = out_dir / DISPLAY_ITEMS_NAME
    display_items.write_text(display_items_markdown(aim2_figure, is_interim), encoding="utf-8")
    docx = out_dir / f"VO2_draft_{date.today().isoformat()}.docx"
    sources = [PAPER_DOCS / name for name in TEXT_FILES] + [display_items, PAPER_DOCS / REFERENCES_FILE]
    command = [pandoc_path(), *map(str, sources), "--from", "markdown", "--to", "docx", "--output", str(docx),
               "--metadata", f"title={DOCUMENT_TITLE}"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"pandoc failed ({result.returncode}): {result.stderr.strip()}")
    if result.stderr.strip():
        logger.warning("pandoc: %s", result.stderr.strip())
    logger.info("Draft for co-authors written to %s", docx)
    return docx
