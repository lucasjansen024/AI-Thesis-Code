"""Display item 4 (paper Table 2): Aim 1 key configurations against WR, descriptive only (no tests; METHODOLOGY §11).

Rows: WR, the carried-forward WR + physiology and no-WR configurations (METHODOLOGY §12), and HR with and without
ANTHRO. Pooled LOSO columns from `A1_pooled/summary/summary.csv` (group `all`): RMSE, its difference to WR (difference
of the pooled means), bias, r_rm, the repeated-measures Bland–Altman 95 % LoA, and the VO₂peak error (ramp visits).
Reference columns: RMSE of the within-cohort runs `A1_gies` and `A1_lucas`.

Files in `outputs/paper/`: `aim1_key_configurations.csv` (unrounded) and `aim1_key_configurations.md`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from figures.aim1_dot_plot import REFERENCE_CONFIG, WITHIN_COHORT_RUN_IDS, read_summary
from figures.style import COHORT_LABELS, PAPER_ROOT
from pipeline.runs import A1_POOLED
from pipeline.train import RUNS_ROOT

VALUES_NAME, TABLE_NAME = "aim1_key_configurations.csv", "aim1_key_configurations.md"
KEY_CONFIGURATIONS = ("WR", "WR+VENT", "HR+VENT+TSI+ANTHRO", "HR", "HR+ANTHRO")
POOLED_COLUMNS = ("n_participants", "rmse_mean", "rmse_sd_participants", "bias_mean", "bias_sd_participants",
                  "r_rm_mean", "ba_loa_lo_mean", "ba_loa_hi_mean", "vo2peak_diff_mean", "vo2peak_diff_sd_participants")
MINUS = "−"


def key_configuration_values(runs_root: Path = RUNS_ROOT) -> pd.DataFrame:
    """One row per key configuration (index config_id): the pooled columns, `rmse_diff_vs_wr`, and per cohort the
    within-cohort `rmse_mean_<cohort>`, `rmse_sd_participants_<cohort>`, `n_participants_<cohort>`."""
    pooled = read_summary(A1_POOLED.run_id, runs_root=runs_root)
    tables = [pooled.loc[list(KEY_CONFIGURATIONS), list(POOLED_COLUMNS)].assign(
        rmse_diff_vs_wr=lambda d: d["rmse_mean"] - pooled.loc[REFERENCE_CONFIG, "rmse_mean"])]
    for cohort, run_id in WITHIN_COHORT_RUN_IDS.items():
        rows = read_summary(run_id, runs_root=runs_root)
        columns = ["rmse_mean", "rmse_sd_participants", "n_participants"]
        tables.append(rows.loc[list(KEY_CONFIGURATIONS), columns].add_suffix(f"_{cohort.lower()}"))
    return pd.concat(tables, axis=1)


def _number(value: float, decimals: int = 0, signed: bool = False) -> str:
    text = f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"
    return text.replace("-", MINUS)


def _mean_sd(mean: float, sd: float, signed: bool = False) -> str:
    return f"{_number(mean, signed=signed)} ± {_number(sd)}"


def _row(config: str, v: pd.Series) -> str:
    diff = "—" if config == REFERENCE_CONFIG else _number(v["rmse_diff_vs_wr"], signed=True)
    cells = [f"`{config}`", _mean_sd(v["rmse_mean"], v["rmse_sd_participants"]), diff,
             _mean_sd(v["bias_mean"], v["bias_sd_participants"], signed=True), _number(v["r_rm_mean"], 3),
             f"{_number(v['ba_loa_lo_mean'])} to {_number(v['ba_loa_hi_mean'])}",
             _mean_sd(v["vo2peak_diff_mean"], v["vo2peak_diff_sd_participants"], signed=True)]
    cells += [_mean_sd(v[f"rmse_mean_{c.lower()}"], v[f"rmse_sd_participants_{c.lower()}"])
              for c in WITHIN_COHORT_RUN_IDS]
    return "| " + " | ".join(cells) + " |"


def format_table(values: pd.DataFrame) -> str:
    """Markdown Table 2: pooled LOSO columns, then the within-cohort RMSE per cohort."""
    n_pooled = int(values["n_participants"].iloc[0])
    cohort_headers = [f"RMSE, {COHORT_LABELS[c]} only (n = {int(values[f'n_participants_{c.lower()}'].iloc[0])})"
                      for c in WITHIN_COHORT_RUN_IDS]
    headers = ["Configuration", "RMSE", "ΔRMSE vs WR", "Bias", "r_rm", "95 % LoA", "VO₂peak error", *cohort_headers]
    lines = ["| " + " | ".join(headers) + " |", "|---" + "|---:" * (len(headers) - 1) + "|"]
    lines += [_row(config, row) for config, row in values.iterrows()]
    note = (f"Values in mL·min⁻¹ (r_rm unitless). Pooled leave-one-subject-out (n = {n_pooled}) unless marked. "
            "RMSE, bias and VO₂peak error: mean ± SD across participants of the seed-averaged values. ΔRMSE: "
            "difference of the mean RMSE to WR. r_rm and the repeated-measures Bland–Altman 95 % limits of "
            "agreement: over all target seconds, mean of 5 seeds. VO₂peak error: predicted − measured highest 30-s "
            "mean, ramp visits only. Within-cohort columns: leave-one-subject-out within that cohort.")
    return "\n".join(lines) + f"\n\n{note}\n"


def write_aim1_key_configurations(out_dir: Path = PAPER_ROOT, runs_root: Path = RUNS_ROOT) -> tuple[Path, Path]:
    """Table 2 → `<out_dir>/aim1_key_configurations.md`, unrounded values → `.csv`."""
    values = key_configuration_values(runs_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    values_path, table_path = out_dir / VALUES_NAME, out_dir / TABLE_NAME
    values.to_csv(values_path)
    table_path.write_text(format_table(values), encoding="utf-8")
    return values_path, table_path
