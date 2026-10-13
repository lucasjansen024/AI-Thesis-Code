"""Display item 2 (paper Figure 1): study design. A = work-rate profiles of the 7 protocols, one participant per
cohort, on a relative intensity scale; B = the training and test data of every analysis (Aim 1, 2a, 2b, 2c).

Panel A reads the ideal work rate (METHODOLOGY §3.4) from the processed cache that `A1_pooled` trained on, never
rebuilt. Relative intensity = WR / GET power below GET and 1 + (WR - GET) / (peak - GET) above it, so GET = 1 and
peak power = 2 on every panel. GET power is the protocol GET (`inventory/reference/get_power.csv`), the value the
xml loads were set from, so GET-level loads land on 1 exactly (the unrounded sheet value differs by ≤ 1 W); peak power
comes from `figures.table1`.
Panel B is a schematic of METHODOLOGY §10 and Appendix B (2a and 2b shown for one left-out condition each).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import SubFigure
from matplotlib.patches import Circle, Rectangle

from figures.style import (BAND, COHORT_COLORS, COHORT_LABELS, DOUBLE_COLUMN_MM, GRID, INK, INK_2, INK_3,
                           LINE_WIDTH_PT, PANEL_LABEL_SIZE_PT, PAPER, PAPER_ROOT, RC_PARAMS, TITLE_SIZE_PT,
                           mm_to_inch, save)
from figures.table1 import read_characteristics
from pipeline.prepare.grid import REST, WARMUP
from pipeline.residuals import cache_work_rate, read_get_power
from pipeline.runs import A1_POOLED
from pipeline.train import RUNS_ROOT

FIGURE_NAME = "study_design"
EXAMPLE_PARTICIPANTS = {"Gies": "G01", "Lucas": "L01"}
PROTOCOLS = {"Gies": ("Ramp", "Easy", "Medium", "Hard"), "Lucas": ("Ramp", "PRBS", "Interval", "Step")}
PROTOCOL_NOTES = {  # METHODOLOGY §2.2
    "Ramp": r"20–35 W·min$^{-1}$ to task failure",
    "Easy": "PRBS 25 W ↔ 90 % GET",
    "Medium": "PRBS 25 W ↔ Δ30 %",
    "Hard": "PRBS GET ↔ Δ30 %",
    "PRBS": "PRBS 25 W ↔ Δ40 %, 30-s bits",
    "Interval": "10 × (2 min 25 W + 1 min Δ70 %)",
    "Step": "80 % GET → Δ30 % → 80 % GET → Δ60 % → 25 W → 60 % GET",
}
TIME_MAX_S = 2250
TIME_TICKS_MIN = (0, 10, 20, 30)
RELATIVE_MAX = 2.15
GET_LEVEL, PEAK_LEVEL = 1.0, 2.0
PROFILE_FILL_ALPHA = 0.12
BAND_LABEL_X_S = 15
PANEL_A_HEIGHT_MM = 92
PANEL_B_HEIGHT_MM = 102
PANEL_A_GRID = {"left": 0.07, "right": 0.98, "top": 0.86, "bottom": 0.11, "hspace": 0.75, "wspace": 0.12}

# Panel B, in mm. Cell states: T train, S train subsampled, X test, B train + test (LOSO), Y subsampled + test,
# "-" unused. Columns follow PROTOCOLS: Gies then Lucas.
MATRIX_ROWS = (
    ("Aim 1 · which inputs", None, None, None),
    ("Pooled LOSO", "BBBBBBBB", "26 → 1", "main analysis"),
    ("Within-cohort LOSO, G", "BBBB----", "13 → 1", "reference for 2c, L → G"),
    ("Within-cohort LOSO, L", "----BBBB", "12 → 1", "reference for 2c, G → L"),
    ("Aim 2a · unseen intensity (example: Hard left out)", None, None, None),
    ("Unseen: Hard left out", "-TTX----", "13 → 1", "the row below, same test data"),
    ("Seen: size-matched", "-SSY----", "13 → 1", ""),
    ("Aim 2b · unseen protocol (example: Step left out)", None, None, None),
    ("Unseen: Step left out", "----TTTX", "12 → 1", "the row below, same test data"),
    ("Seen: size-matched", "----SSSY", "12 → 1", ""),
    ("Aim 2c · unseen cohort", None, None, None),
    ("G → L", "TTTTXXXX", "14 → 13", "within-cohort LOSO, L"),
    ("L → G", "XXXXTTTT", "13 → 14", "within-cohort LOSO, G"),
)
TRAIN_STATES, SUBSAMPLED_STATES, TEST_STATES = frozenset("TB"), frozenset("SY"), frozenset("XBY")
CELL_MM, CELL_GAP_MM, COHORT_GAP_MM = 4.4, 1.2, 3.0
MATRIX_X0_MM, LABEL_X_MM, ROW_INDENT_MM = 62.0, 6.0, 3.0
N_COLUMN_X_MM, COMPARE_X_MM = 120.0, 140.0
HEADER_MM, ROW_MM, GROUP_MM = 20.0, 5.4, 5.6
COHORT_HEADER_Y_MM = 8.0
SUBSAMPLED_ALPHA = 0.3
HATCH = "//////"
TEST_DOT_MM, TEST_RING_MM = 0.9, 1.3
LEGEND_TEXT_GAP_MM = 1.5
LEGEND_ITEMS = (  # x (mm), cell state, colour, label; x positions checked on the rendered figure
    (LABEL_X_MM, "T", COHORT_COLORS["Gies"], f"{COHORT_LABELS['Gies']} training data"),
    (44.0, "T", COHORT_COLORS["Lucas"], f"{COHORT_LABELS['Lucas']} training data"),
    (82.0, "S", INK_3, "training data, subsampled to the unseen arm's size"),
    (148.0, "X", INK_3, "test data"),
    (168.0, "-", INK_3, "not used"),
)


def relative_intensity(wr_w: np.ndarray, get_w: float, peak_w: float) -> np.ndarray:
    """GET = 1, peak power = 2: below GET a fraction of GET power, above it 1 + the Δ fraction (in power)."""
    if not 0 < get_w < peak_w:
        raise ValueError(f"need 0 < GET ({get_w} W) < peak power ({peak_w} W)")
    return np.where(wr_w <= get_w, wr_w / get_w, GET_LEVEL + (wr_w - get_w) / (peak_w - get_w))


def example_work_rates(run_id: str = A1_POOLED.run_id, runs_root: Path = RUNS_ROOT) -> pd.DataFrame:
    """`visit_id, t_s, phase_code, wr` for the example participants' visits, from the run's training cache."""
    wanted = pd.DataFrame([{"fold": pid, "visit_id": f"{pid}_{protocol}"}
                           for cohort, pid in EXAMPLE_PARTICIPANTS.items() for protocol in PROTOCOLS[cohort]])
    return cache_work_rate(run_id, wanted, runs_root=runs_root)


def _phase_span(visit: pd.DataFrame, phase_code: int) -> tuple[float, float]:
    seconds = visit.loc[visit["phase_code"] == phase_code, "t_s"]
    if seconds.empty:
        raise ValueError(f"{visit['visit_id'].iloc[0]}: no seconds of phase {phase_code} in the cache")
    return float(seconds.min()), float(seconds.max()) + 1


def _profile_axes(ax: Axes, visit: pd.DataFrame, protocol: str, color: str, is_band_labelled: bool) -> None:
    for phase_code in (REST, WARMUP):
        start, stop = _phase_span(visit, phase_code)
        ax.axvspan(start, stop, color=BAND, linewidth=0, zorder=0)
    if is_band_labelled:  # the 2-min rest band is too narrow for its own label
        ax.text(BAND_LABEL_X_S, RELATIVE_MAX * 0.97, "rest, warm-up", ha="left", va="top", color=INK_3)
    for level in (GET_LEVEL, PEAK_LEVEL):
        ax.axhline(level, color=GRID, linewidth=0.5, zorder=0)
    t, rel = visit["t_s"].to_numpy(), visit["relative"].to_numpy()
    ax.fill_between(t, rel, step="post", color=color, alpha=PROFILE_FILL_ALPHA, linewidth=0)
    ax.plot(t, rel, drawstyle="steps-post", color=color, linewidth=LINE_WIDTH_PT)
    title = ax.text(0, 1.06, protocol, transform=ax.transAxes, weight="bold", va="bottom")
    ax.annotate(PROTOCOL_NOTES[protocol], xy=(1, 0), xycoords=title, xytext=(4, 0), textcoords="offset points",
                va="bottom", color=INK_2)
    ax.set_xlim(0, TIME_MAX_S)
    ax.set_ylim(0, RELATIVE_MAX)
    ax.set_xticks([m * 60 for m in TIME_TICKS_MIN], [str(m) for m in TIME_TICKS_MIN])
    ax.set_yticks([0, GET_LEVEL, PEAK_LEVEL], ["0 W", "GET", "peak"])
    ax.spines[["top", "right"]].set_visible(False)


def _draw_panel_a(subfig: SubFigure, work_rates: pd.DataFrame, characteristics: pd.DataFrame,
                  get_power: pd.DataFrame) -> None:
    axes = subfig.subplots(4, 2, sharex=True, sharey=True, gridspec_kw=PANEL_A_GRID)
    for col, (cohort, pid) in enumerate(EXAMPLE_PARTICIPANTS.items()):
        n = int((characteristics["cohort"] == cohort).sum())
        axes[0, col].text(0, 1.55, f"{COHORT_LABELS[cohort]} (n = {n}), participant {pid}",
                          transform=axes[0, col].transAxes, weight="bold", size=TITLE_SIZE_PT)
        for row, protocol in enumerate(PROTOCOLS[cohort]):
            visit = work_rates[work_rates["visit_id"] == f"{pid}_{protocol}"]
            if visit.empty:
                raise ValueError(f"{pid}_{protocol}: not in the cache")
            relative = relative_intensity(visit["wr"].to_numpy(), get_power.at[pid, "get_w"],
                                          characteristics.at[pid, "peak_power_w"])
            _profile_axes(axes[row, col], visit.assign(relative=relative), protocol, COHORT_COLORS[cohort],
                          is_band_labelled=(row == 0 and col == 0))
        axes[-1, col].set_xlabel("time (min)")
    subfig.text(0.005, 0.995, "A", weight="bold", size=PANEL_LABEL_SIZE_PT, va="top")


def _column_x(index: int) -> float:
    cohort_gap = COHORT_GAP_MM if index >= len(PROTOCOLS["Gies"]) else 0.0
    return MATRIX_X0_MM + index * (CELL_MM + CELL_GAP_MM) + cohort_gap


def _cell(ax: Axes, x: float, y: float, state: str, color: str) -> None:
    if state in TRAIN_STATES:
        ax.add_patch(Rectangle((x, y), CELL_MM, CELL_MM, facecolor=color, linewidth=0))
    elif state in SUBSAMPLED_STATES:
        ax.add_patch(Rectangle((x, y), CELL_MM, CELL_MM, facecolor=color, alpha=SUBSAMPLED_ALPHA, linewidth=0))
        ax.add_patch(Rectangle((x, y), CELL_MM, CELL_MM, fill=False, hatch=HATCH, edgecolor=INK_2, linewidth=0))
    else:
        ax.add_patch(Rectangle((x, y), CELL_MM, CELL_MM, facecolor=BAND, linewidth=0))
    if state in TEST_STATES:
        center = (x + CELL_MM / 2, y + CELL_MM / 2)
        ax.add_patch(Circle(center, TEST_RING_MM, facecolor=PAPER, linewidth=0))
        ax.add_patch(Circle(center, TEST_DOT_MM, facecolor=INK, linewidth=0))


def _matrix_header(ax: Axes) -> None:
    columns = [(cohort, protocol) for cohort in PROTOCOLS for protocol in PROTOCOLS[cohort]]
    for cohort in PROTOCOLS:
        first = next(i for i, (c, _) in enumerate(columns) if c == cohort)
        ax.text(_column_x(first), COHORT_HEADER_Y_MM, COHORT_LABELS[cohort], weight="bold", va="center")
    for i, (_, protocol) in enumerate(columns):
        ax.text(_column_x(i) + CELL_MM / 2, HEADER_MM - 1.5, protocol, rotation=45, ha="left", va="bottom",
                rotation_mode="anchor", color=INK_2)
    ax.text(N_COLUMN_X_MM, HEADER_MM - 1.5, "participants\ntrain → test", va="bottom", color=INK_2)
    ax.text(COMPARE_X_MM, HEADER_MM - 1.5, "compared with", va="bottom", color=INK_2)


def _matrix_legend(ax: Axes, y: float) -> None:
    for x, state, color, label in LEGEND_ITEMS:
        _cell(ax, x, y, state, color)
        ax.text(x + CELL_MM + LEGEND_TEXT_GAP_MM, y + CELL_MM / 2, label, va="center", color=INK_2)


def _draw_panel_b(subfig: SubFigure) -> None:
    ax = subfig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, DOUBLE_COLUMN_MM)
    ax.set_ylim(PANEL_B_HEIGHT_MM, 0)
    ax.set_axis_off()
    _matrix_header(ax)
    y = HEADER_MM
    for label, cells, n, compared in MATRIX_ROWS:
        if cells is None:
            ax.axhline(y + 0.6, xmin=LABEL_X_MM / DOUBLE_COLUMN_MM, color=GRID, linewidth=0.5)
            ax.text(LABEL_X_MM, y + GROUP_MM / 2 + 0.6, label, weight="bold", va="center")
            y += GROUP_MM
            continue
        ax.text(LABEL_X_MM + ROW_INDENT_MM, y + CELL_MM / 2, label, va="center")
        for i, state in enumerate(cells):
            cohort = "Gies" if i < len(PROTOCOLS["Gies"]) else "Lucas"
            _cell(ax, _column_x(i), y, state, COHORT_COLORS[cohort])
        ax.text(N_COLUMN_X_MM, y + CELL_MM / 2, n, va="center")
        ax.text(COMPARE_X_MM, y + CELL_MM / 2, compared, va="center", color=INK_2)
        y += ROW_MM
    _matrix_legend(ax, y + 3)
    subfig.text(0.005, 0.995, "B", weight="bold", size=PANEL_LABEL_SIZE_PT, va="top")


def write_study_design(out_dir: Path = PAPER_ROOT, runs_root: Path = RUNS_ROOT) -> tuple[Path, Path]:
    """Figure → `<out_dir>/study_design.{pdf,png}`."""
    characteristics = read_characteristics()
    work_rates = example_work_rates(runs_root=runs_root)
    size = (mm_to_inch(DOUBLE_COLUMN_MM), mm_to_inch(PANEL_A_HEIGHT_MM + PANEL_B_HEIGHT_MM))
    with plt.rc_context({**RC_PARAMS, "hatch.linewidth": 0.5}):
        fig = plt.figure(figsize=size)
        panel_a, panel_b = fig.subfigures(2, 1, height_ratios=(PANEL_A_HEIGHT_MM, PANEL_B_HEIGHT_MM))
        _draw_panel_a(panel_a, work_rates, characteristics, read_get_power())
        _draw_panel_b(panel_b)
        paths = save(fig, FIGURE_NAME, out_dir)
        plt.close(fig)
    return paths
