"""Display item 3 (paper Figure 2): Aim 1 RMSE per input configuration. Pooled LOSO as mean ± SD across participants,
the within-cohort LOSO runs as small markers, and a reference line at WR alone (METHODOLOGY §11.1).

Configurations: the single inputs with and without ANTHRO, WR plus one physiological domain, and the best configuration
without WR (carry-forward rule, METHODOLOGY §12). The main text uses TSI as the NIRS variable; the HBDIFF
configurations go to the supplement. Values are read from each run's `summary/summary.csv`, group `all`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from figures.style import (COHORT_COLORS, COHORT_LABELS, GRID, INK, INK_2, INK_3, PAPER, PAPER_ROOT, RC_PARAMS,
                           SINGLE_COLUMN_MM, mm_to_inch, save)
from pipeline.evaluate import ALL_GROUP, SUMMARY_NAME, SUMMARY_TABLE_NAME
from pipeline.runs import A1_GIES, A1_LUCAS, A1_POOLED
from pipeline.train import RUNS_ROOT

FIGURE_NAME = "aim1_dot_plot"
REFERENCE_CONFIG = "WR"
CONFIG_GROUPS = (
    ("Single input", ("WR", "HR", "VENT", "TSI")),
    ("Single input + ANTHRO", ("WR+ANTHRO", "HR+ANTHRO", "VENT+ANTHRO", "TSI+ANTHRO")),
    ("WR + one domain", ("WR+HR", "WR+VENT", "WR+TSI")),
    ("Best without WR", ("HR+VENT+TSI+ANTHRO",)),
)
WITHIN_COHORT_RUN_IDS = {"Gies": A1_GIES.run_id, "Lucas": A1_LUCAS.run_id}
COHORT_MARKERS = {"Gies": "D", "Lucas": "s"}
COHORT_OFFSET_ROWS = {"Gies": -0.26, "Lucas": 0.26}  # the y axis is inverted, so Cohort G sits above the pooled dot
POOLED_MARKER_PT, COHORT_MARKER_PT = 4.0, 2.6
MARKER_RING_PT = 0.5
HEADER_ROWS = 1.3  # space a group heading takes above its first row
HEADING_OFFSET_ROWS = 0.45  # heading centre below the top of its header space
GROUP_GAP_ROWS = 0.4  # extra space above every heading but the first
HEADING_X_FIG, LABEL_X_FIG = 0.02, 0.05  # labels are indented under their heading
REFERENCE_LABEL_GAP = 8  # mL·min⁻¹ between the WR line and its label
X_MIN, X_MAX = 100, 700  # a dot plot needs no zero; the lowest whisker is ≈ 156
X_TICK_STEP = 100
ROW_HEIGHT_MM = 4.6
MARGIN_HEIGHT_MM = 15
AXES_GRID = {"left": 0.38, "right": 0.97, "top": 0.92, "bottom": 0.12}


def read_summary(run_id: str, group: str = ALL_GROUP, runs_root: Path = RUNS_ROOT) -> pd.DataFrame:
    """One run's `summary.csv` rows for `group`, indexed by config_id."""
    summary = pd.read_csv(runs_root / run_id / SUMMARY_NAME / SUMMARY_TABLE_NAME)
    rows = summary[summary["group"] == group].set_index("config_id")
    if rows.empty:
        raise ValueError(f"{run_id}: no summary rows for group {group!r}")
    return rows


def _configurations() -> tuple[str, ...]:
    return tuple(config for _, configs in CONFIG_GROUPS for config in configs)


def _check_configurations(rows: pd.DataFrame, run_id: str) -> None:
    missing = sorted(set(_configurations()) - set(rows.index))
    if missing:
        raise ValueError(f"{run_id}: summary has no rows for {missing}")


def _row_positions() -> tuple[dict[str, float], list[tuple[str, float]]]:
    """y position per configuration, and (heading, y) per group; a heading takes HEADER_ROWS above its group."""
    positions, headings, y = {}, [], 0.0
    for index, (heading, configs) in enumerate(CONFIG_GROUPS):
        y += GROUP_GAP_ROWS if index else 0.0
        headings.append((heading, y + HEADING_OFFSET_ROWS))
        y += HEADER_ROWS
        for config in configs:
            positions[config] = y
            y += 1.0
    return positions, headings


def _draw(ax: Axes, pooled: pd.DataFrame, within: dict[str, pd.DataFrame]) -> None:
    positions, headings = _row_positions()
    configs = list(positions)
    ys = [positions[c] for c in configs]
    reference = pooled.loc[REFERENCE_CONFIG, "rmse_mean"]
    ax.axvline(reference, color=INK_3, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.text(reference + REFERENCE_LABEL_GAP, -0.5, f"WR alone, {reference:.0f}", color=INK_2, ha="left",
            va="center")
    ax.errorbar(pooled.loc[configs, "rmse_mean"], ys, xerr=pooled.loc[configs, "rmse_sd_participants"], fmt="o",
                color=INK, ecolor=INK_2, elinewidth=0.8, capsize=0, markersize=POOLED_MARKER_PT,
                markeredgecolor=PAPER, markeredgewidth=MARKER_RING_PT, zorder=3)
    for cohort, rows in within.items():
        ax.plot(rows.loc[configs, "rmse_mean"], [y + COHORT_OFFSET_ROWS[cohort] for y in ys], linestyle="none",
                marker=COHORT_MARKERS[cohort], markersize=COHORT_MARKER_PT, color=COHORT_COLORS[cohort],
                markeredgecolor=PAPER, markeredgewidth=MARKER_RING_PT, zorder=2)
    label_x = blended_transform_factory(ax.figure.transFigure, ax.transData)  # header rows hold no data
    for heading, y in headings:
        ax.text(HEADING_X_FIG, y, heading, transform=label_x, ha="left", va="center", color=INK, fontweight="bold")
    for config, y in positions.items():
        ax.text(LABEL_X_FIG, y, config, transform=label_x, ha="left", va="center", color=INK_2)
    ax.set_yticks([])
    ax.set_ylim(max(ys) + 0.7, -1.0)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks(range(X_MIN, X_MAX + 1, X_TICK_STEP))
    ax.set_xlabel(r"RMSE (mL·min$^{-1}$)")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def _legend_handles() -> list[Line2D]:
    pooled = Line2D([], [], marker="o", color=INK, markersize=POOLED_MARKER_PT, linewidth=0.8,
                    label="Pooled, mean ± SD")
    cohorts = [Line2D([], [], marker=COHORT_MARKERS[c], color=COHORT_COLORS[c], markersize=COHORT_MARKER_PT,
                      linestyle="none", label=f"{COHORT_LABELS[c]} only") for c in COHORT_MARKERS]
    return [pooled, *cohorts]


def write_aim1_dot_plot(out_dir: Path = PAPER_ROOT, runs_root: Path = RUNS_ROOT) -> tuple[Path, Path]:
    """Figure → `<out_dir>/aim1_dot_plot.{pdf,png}`."""
    pooled = read_summary(A1_POOLED.run_id, runs_root=runs_root)
    within = {cohort: read_summary(run_id, runs_root=runs_root) for cohort, run_id in WITHIN_COHORT_RUN_IDS.items()}
    for run_id, rows in ((A1_POOLED.run_id, pooled), *((WITHIN_COHORT_RUN_IDS[c], r) for c, r in within.items())):
        _check_configurations(rows, run_id)
    n_rows = (len(_configurations()) + HEADER_ROWS * len(CONFIG_GROUPS) + GROUP_GAP_ROWS * (len(CONFIG_GROUPS) - 1))
    size = (mm_to_inch(SINGLE_COLUMN_MM), mm_to_inch(n_rows * ROW_HEIGHT_MM + MARGIN_HEIGHT_MM))
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=size, gridspec_kw=AXES_GRID)
        _draw(ax, pooled, within)
        fig.legend(handles=_legend_handles(), loc="upper left", bbox_to_anchor=(HEADING_X_FIG, 0.995), frameon=False,
                   ncol=len(COHORT_MARKERS) + 1, handletextpad=0.4, columnspacing=1.6, borderaxespad=0,
                   labelcolor=INK_2)
        paths = save(fig, FIGURE_NAME, out_dir)
        plt.close(fig)
    return paths
