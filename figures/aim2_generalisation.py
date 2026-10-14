"""Display item 5 (paper Figure 3): Aim 2 generalisation. Per test condition and carried-forward configuration, the
mean participant RMSE with the condition seen vs unseen in training, joined as a dumbbell, and the mean difference
unseen − seen with its 95 % t-CI as text (METHODOLOGY Appendix B.6).

Panels: 2a (one PRBS level left out, Cohort G), 2b (one protocol left out, Cohort L) and 2c (trained on the other
cohort vs the test cohort's within-cohort LOSO, all visits). Rows are grouped by configuration; every panel uses the
same RMSE scale. Values come from the Aim 2 `summary.csv`; conditions it does not hold yet are skipped with a warning,
so the script also runs on the smoke and 2c-only outputs.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from figures.style import (GRID, INK, INK_2, INK_3, PANEL_LABEL_SIZE_PT, PAPER, PAPER_ROOT, RC_PARAMS,
                           SINGLE_COLUMN_MM, TITLE_SIZE_PT, mm_to_inch, save)
from pipeline.aim2_outcome import ALL_PROTOCOLS, OUTCOME_ROOT, SUMMARY_TABLE_NAME
from pipeline.runs import A2_CONFIGURATIONS, A2A_POOL, A2A_STUDY, A2B_POOL, A2B_STUDY, A2C_STUDY

logger = logging.getLogger(__name__)

FIGURE_NAME = "aim2_generalisation"
TRANSFER_LABELS = {"Gies_to_Lucas": "G → L", "Lucas_to_Gies": "L → G"}  # 2c condition names in aim2_outcome
PANELS = (
    (A2A_STUDY, "PRBS level left out (Cohort G)", A2A_POOL),
    (A2B_STUDY, "Protocol left out (Cohort L)", A2B_POOL),
    (A2C_STUDY, "Cohort transfer, all visits", tuple(TRANSFER_LABELS)),
)
MARKER_PT, MARKER_EDGE_PT, MARKER_RING_PT = 4.0, 0.8, 0.5
CONNECTOR_PT = 0.8
TITLE_ROWS = 1.4  # panel title row above the first heading
HEADER_ROWS = 1.1  # space a configuration heading takes above its first row
HEADING_OFFSET_ROWS = 0.4  # heading centre below the top of its header space
GROUP_GAP_ROWS = 0.3  # extra space above every heading but the first
HEADING_X_FIG, LABEL_X_FIG, DIFF_X_FIG = 0.02, 0.06, 0.99  # headings run over the axes; their rows hold no data
X_TICK_STEP = 100
ROW_HEIGHT_MM = 3.8
TOP_MM, BOTTOM_MM, PANEL_GAP_MM = 6.0, 9.0, 7.0
AXES_LEFT, AXES_RIGHT = 0.22, 0.72
UNICODE_MINUS = "−"


def read_outcome(path: Path) -> pd.DataFrame:
    """The plotted rows of an Aim 2 `summary.csv`: 2a/2b per left-out condition, 2c on all visits."""
    summary = pd.read_csv(path)
    is_transfer = summary["study"] == A2C_STUDY
    plotted = summary[(is_transfer & (summary["scope"] == ALL_PROTOCOLS))
                      | (~is_transfer & (summary["scope"] == summary["condition"]))]
    if plotted.empty:
        raise ValueError(f"{path}: no Aim 2 rows to plot")
    return plotted.set_index(["study", "condition", "config_id"])


def _present_conditions(outcome: pd.DataFrame, study: str, conditions: tuple[str, ...]) -> tuple[str, ...]:
    """The conditions with a row for every configuration; logs what is missing."""
    present = tuple(c for c in conditions
                    if all((study, c, config) in outcome.index for config in A2_CONFIGURATIONS))
    missing = [c for c in conditions if c not in present]
    if missing:
        logger.warning("Aim 2 figure: study %s has no complete rows for %s; skipped", study, missing)
    return present


def _row_positions(conditions: tuple[str, ...]) -> tuple[dict[tuple[str, str], float], list[tuple[str, float]]]:
    """y per (configuration, condition), and (heading, y) per configuration; the panel title sits at y = 0."""
    positions, headings, y = {}, [], TITLE_ROWS
    for index, config in enumerate(A2_CONFIGURATIONS):
        y += GROUP_GAP_ROWS if index else 0.0
        headings.append((config, y + HEADING_OFFSET_ROWS))
        y += HEADER_ROWS
        for condition in conditions:
            positions[(config, condition)] = y
            y += 1.0
    return positions, headings


def _panel_rows(n_conditions: int) -> float:
    return TITLE_ROWS + len(A2_CONFIGURATIONS) * (HEADER_ROWS + n_conditions) + GROUP_GAP_ROWS * (
        len(A2_CONFIGURATIONS) - 1)


def _signed(value: float) -> str:
    rounded = int(round(value))
    return f"{rounded:+d}".replace("-", UNICODE_MINUS) if rounded else "0"


def _diff_text(row: pd.Series) -> str:
    """`+53 [−81, 186]`; the CI is left out when it is undefined (one participant)."""
    if math.isnan(row["diff_ci_lo"]):
        return _signed(row["diff_mean"])
    lo, hi = (f"{int(round(row[c])):d}".replace("-", UNICODE_MINUS) for c in ("diff_ci_lo", "diff_ci_hi"))
    return f"{_signed(row['diff_mean'])} [{lo}, {hi}]"


def _condition_label(study: str, condition: str) -> str:
    return TRANSFER_LABELS.get(condition, condition) if study == A2C_STUDY else condition


def _draw_marks(ax: Axes, rows: pd.DataFrame, ys: list[float]) -> None:
    for seen, unseen, y in zip(rows["rmse_seen_mean"], rows["rmse_unseen_mean"], ys):
        ax.plot([seen, unseen], [y, y], color=INK_3, linewidth=CONNECTOR_PT, solid_capstyle="butt", zorder=1)
    ax.plot(rows["rmse_seen_mean"], ys, linestyle="none", marker="o", markersize=MARKER_PT,
            markerfacecolor=PAPER, markeredgecolor=INK, markeredgewidth=MARKER_EDGE_PT, zorder=2)
    ax.plot(rows["rmse_unseen_mean"], ys, linestyle="none", marker="o", markersize=MARKER_PT, color=INK,
            markeredgecolor=PAPER, markeredgewidth=MARKER_RING_PT, zorder=3)


def _draw_panel(ax: Axes, outcome: pd.DataFrame, study: str, title: str, conditions: tuple[str, ...],
                letter: str) -> None:
    positions, headings = _row_positions(conditions)
    keys = [(study, condition, config) for config, condition in positions]
    rows, ys = outcome.loc[keys], list(positions.values())
    _draw_marks(ax, rows, ys)
    label_x = blended_transform_factory(ax.figure.transFigure, ax.transData)
    ax.text(HEADING_X_FIG, 0.0, letter, transform=label_x, ha="left", va="center", weight="bold",
            size=PANEL_LABEL_SIZE_PT)
    ax.text(LABEL_X_FIG, 0.0, f"{study}  {title}", transform=label_x, ha="left", va="center", weight="bold",
            size=TITLE_SIZE_PT)
    ax.text(DIFF_X_FIG, 0.0, "Δ [95 % CI]", transform=label_x, ha="right", va="center", color=INK_2)
    for heading, y in headings:
        ax.text(HEADING_X_FIG, y, heading, transform=label_x, ha="left", va="center", color=INK, fontweight="bold")
    for ((config, condition), y), (_, row) in zip(positions.items(), rows.iterrows()):
        ax.text(LABEL_X_FIG, y, _condition_label(study, condition), transform=label_x, ha="left", va="center",
                color=INK_2)
        ax.text(DIFF_X_FIG, y, _diff_text(row), transform=label_x, ha="right", va="center", color=INK_2)
    ax.set_yticks([])
    ax.set_ylim(max(ys) + 0.7, -0.6)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def _x_limits(outcome: pd.DataFrame) -> tuple[int, int]:
    """Shared RMSE scale: the plotted means rounded out to the tick step (a dot plot needs no zero)."""
    values = outcome[["rmse_seen_mean", "rmse_unseen_mean"]].to_numpy()
    return (int(math.floor(values.min() / X_TICK_STEP) * X_TICK_STEP),
            int(math.ceil(values.max() / X_TICK_STEP) * X_TICK_STEP))


def _legend_handles() -> list[Line2D]:
    seen = Line2D([], [], marker="o", linestyle="none", markersize=MARKER_PT, markerfacecolor=PAPER,
                  markeredgecolor=INK, markeredgewidth=MARKER_EDGE_PT, label="Seen in training")
    unseen = Line2D([], [], marker="o", linestyle="none", markersize=MARKER_PT, color=INK,
                    label="Unseen (left out / other cohort)")
    return [seen, unseen]


def write_aim2_generalisation(summary_path: Path = OUTCOME_ROOT / SUMMARY_TABLE_NAME,
                              out_dir: Path = PAPER_ROOT) -> tuple[Path, Path]:
    """Figure → `<out_dir>/aim2_generalisation.{pdf,png}`."""
    outcome = read_outcome(summary_path)
    panels = [(study, title, present) for study, title, conditions in PANELS
              if (present := _present_conditions(outcome, study, conditions))]
    if not panels:
        raise ValueError(f"{summary_path}: no study has complete rows for {A2_CONFIGURATIONS}")
    heights_mm = [_panel_rows(len(conditions)) * ROW_HEIGHT_MM for _, _, conditions in panels]
    height_mm = sum(heights_mm) + PANEL_GAP_MM * (len(panels) - 1) + TOP_MM + BOTTOM_MM
    grid = {"height_ratios": heights_mm, "hspace": PANEL_GAP_MM / (sum(heights_mm) / len(heights_mm)),
            "left": AXES_LEFT, "right": AXES_RIGHT, "top": 1 - TOP_MM / height_mm, "bottom": BOTTOM_MM / height_mm}
    plotted = outcome.loc[[(study, condition, config) for study, _, conditions in panels
                           for condition in conditions for config in A2_CONFIGURATIONS]]
    x_min, x_max = _x_limits(plotted)
    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(len(panels), 1, figsize=(mm_to_inch(SINGLE_COLUMN_MM), mm_to_inch(height_mm)),
                                 gridspec_kw=grid, squeeze=False)
        for ax, (study, title, conditions), letter in zip(axes[:, 0], panels, "ABC"):
            _draw_panel(ax, outcome, study, title, conditions, letter)
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(range(x_min, x_max + 1, X_TICK_STEP))
        axes[-1, 0].set_xlabel(r"RMSE (mL·min$^{-1}$)")
        fig.legend(handles=_legend_handles(), loc="upper left", bbox_to_anchor=(HEADING_X_FIG, 0.998), frameon=False,
                   ncol=2, handletextpad=0.4, columnspacing=1.6, borderaxespad=0, labelcolor=INK_2)
        paths = save(fig, FIGURE_NAME, out_dir)
        plt.close(fig)
    return paths
