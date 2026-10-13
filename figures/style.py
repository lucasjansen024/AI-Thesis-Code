"""Shared look of the paper's display items: cohort colours, ink, journal widths, fonts, and saving.

Colours are the dataviz reference palette (light mode): categorical slots 1-2 for the cohorts, which pass the
colour-vision checks as a pair. Figures are print plates on white, so there is no dark variant.
"""
from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from pipeline.provenance import REPO_ROOT

PAPER_ROOT = REPO_ROOT / "outputs" / "paper"

COHORT_COLORS = {"Gies": "#2a78d6", "Lucas": "#eb6834"}
COHORT_LABELS = {"Gies": "Cohort G", "Lucas": "Cohort L"}
INK = "#141414"
INK_2 = "#555555"
INK_3 = "#8c8c8c"
GRID = "#e4e4e4"
BAND = "#f1f1f1"
PAPER = "#ffffff"

MM_PER_INCH = 25.4
DOUBLE_COLUMN_MM = 190  # Elsevier full page width
FONT_SIZE_PT = 7
TITLE_SIZE_PT = 8
PANEL_LABEL_SIZE_PT = 10
LINE_WIDTH_PT = 1.0
PNG_DPI = 600

RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FONT_SIZE_PT,
    "axes.titlesize": FONT_SIZE_PT,
    "axes.labelsize": FONT_SIZE_PT,
    "axes.edgecolor": INK_3,
    "axes.labelcolor": INK_2,
    "axes.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "text.color": INK,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "pdf.fonttype": 42,  # embed TrueType so the journal can edit text
}


def mm_to_inch(mm: float) -> float:
    return mm / MM_PER_INCH


def save(fig: Figure, name: str, out_dir: Path = PAPER_ROOT) -> tuple[Path, Path]:
    """Write `<name>.pdf` (vector, for submission) and `<name>.png` (for drafts) to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / f"{name}.pdf", out_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=PNG_DPI)
    return pdf, png
