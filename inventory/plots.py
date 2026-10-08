"""One multi-panel PNG per visit on the CPET timeline, for visual checks of loading and alignment."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from inventory.sensors import FIT_FACTOR, NIRS_CHANNELS, TSI, VisitData

PANEL_HEIGHT_IN = 2.2
FIGURE_WIDTH_IN = 16
X_MARGIN_S = 60
FIT_FACTOR_YLIM = (98.0, 100.05)
FIT_FACTOR_LINES = (99.0, 99.9)
DPI = 110
PANELS = ("Work rate (W)", "VO2 (mL/min)", "VE", "HR (bpm)", "BR (1/min)",
          *(f"NIRS Tx{ch} (µM)" for ch in NIRS_CHANNELS), "TSI (%) / fit factor")


def _phase_lines(ax: Axes, visit: VisitData) -> None:
    timeline = visit.timeline
    for x in (0, timeline.rest_end_s, timeline.warmup_end_s, timeline.exercise_end_s):
        if x is not None:
            ax.axvline(x, color="red", linestyle="--", linewidth=0.8)
    for x in timeline.repetition_starts_s[1:]:
        ax.axvline(x, color="red", linestyle=":", linewidth=1.0)
    for x in visit.cpet_phase_starts.values():
        if not math.isnan(x):
            ax.axvline(x, color="orange", linewidth=0.6)


def _plot_wr(ax: Axes, visit: VisitData) -> None:
    if visit.protocol.ok:
        ax.step(visit.protocol.data["t"], visit.protocol.data["W"], where="post", color="black", label="protocol (xml; ramp: CPET protocol name)")
    if visit.reference_wr.ok:
        ref = visit.reference_wr.data
        ax.plot(ref["t"], ref["W"], color="grey", linestyle="--", label="legacy prepared WR")
    if visit.cpet.ok and "Power" in visit.cpet.data.df.columns:
        df = visit.cpet.data.df
        ax.plot(df["t"], df["Power"], ".", markersize=2, color="tab:blue", label="measured power")


def _plot_cpet_and_tymewear(ax: Axes, visit: VisitData, cpet_col: str, tw_col: str | None) -> None:
    if visit.cpet.ok and cpet_col in visit.cpet.data.df.columns:
        df = visit.cpet.data.df
        ax.plot(df["t"], df[cpet_col], ".-", markersize=2, linewidth=0.5, color="tab:blue", label=f"CPET {cpet_col}")
    if tw_col is None or not visit.tymewear.ok:
        return
    series = visit.tymewear.data.series
    # VE units differ between devices, so Tymewear VE gets its own axis.
    target = ax.twinx() if tw_col == "VE" else ax
    target.plot(series["t"], series[tw_col], color="tab:orange", linewidth=0.8, label=f"Tymewear {tw_col}")
    if target is not ax:
        target.set_ylabel("Tymewear VE", color="tab:orange")


def _plot_nirs_channel(ax: Axes, visit: VisitData, channel: int) -> None:
    if not visit.nirs.ok:
        return
    df = visit.nirs.data.df
    for kind, color in (("O2Hb", "red"), ("HHb", "blue"), ("tHb", "green")):
        ax.plot(df["t"], df[f"{kind}{channel}"], color=color, linewidth=0.5, label=kind)


def _plot_tsi(ax: Axes, visit: VisitData) -> None:
    if not visit.nirs.ok:
        return
    df = visit.nirs.data.df
    ax.plot(df["t"], df[TSI], color="green", linewidth=0.5, label="TSI")
    ff_ax = ax.twinx()
    ff_ax.plot(df["t"], df[FIT_FACTOR], color="purple", linewidth=0.5)
    for y in FIT_FACTOR_LINES:
        ff_ax.axhline(y, color="purple", linestyle=":", linewidth=0.8)
    ff_ax.set_ylim(*FIT_FACTOR_YLIM)
    ff_ax.set_ylabel("fit factor", color="purple")


def _title(visit: VisitData) -> str:
    ref, ann = visit.ref, visit.annotation
    files = ", ".join(
        f"{name}={loaded.choice.selected.name if loaded.choice.selected else '—'}"
        for name, loaded in (("CPET", visit.cpet), ("TW", visit.tymewear), ("NIRS", visit.nirs))
    )
    annotation = (f"VE_lag={ann.tymewear_lag_s:g}, HR={ann.hr_source}, NIRS={ann.nirs_repetition}"
                  if ann.present else "no annotation")
    notes = f" | {ann.notes}" if ann.notes else ""
    return f"{ref.visit_id} ({ref.cohort} {ref.source_participant}/{ref.source_visit}, {ref.role})\n{files}\n{annotation}{notes}"


def plot_visit(visit: VisitData, out_dir: Path) -> Path:
    fig, axes = plt.subplots(len(PANELS), 1, sharex=True, figsize=(FIGURE_WIDTH_IN, PANEL_HEIGHT_IN * len(PANELS)))
    _plot_wr(axes[0], visit)
    _plot_cpet_and_tymewear(axes[1], visit, "VO2", None)
    _plot_cpet_and_tymewear(axes[2], visit, "VE", "VE")
    _plot_cpet_and_tymewear(axes[3], visit, "HR", "HR")
    _plot_cpet_and_tymewear(axes[4], visit, "Rf", "BR")
    for ax, channel in zip(axes[5:8], NIRS_CHANNELS):
        _plot_nirs_channel(ax, visit, channel)
    _plot_tsi(axes[8], visit)
    for ax, label in zip(axes, PANELS):
        _phase_lines(ax, visit)
        ax.set_ylabel(label)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=7)
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="grey")
    end = visit.timeline.exercise_end_s or visit.timeline.warmup_end_s
    axes[-1].set_xlim(-X_MARGIN_S, end + X_MARGIN_S)
    axes[-1].set_xlabel("CPET time (s); red dashed = nominal phases, red dotted = PRBS repetition, orange = CPET phase marks")
    fig.suptitle(_title(visit), fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{visit.ref.visit_id}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
