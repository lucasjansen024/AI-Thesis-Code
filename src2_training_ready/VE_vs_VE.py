import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_PREPARED_DIR = Path(r"C:\Users\van Wijk\Documents\AI Master\Codes\data\prepared")
OUTPUT_DIR = Path(r"C:\Users\van Wijk\Documents\AI Master\Codes\data\VE-VE")

# False = gebruik "VE breath by breath"
# True  = gebruik "VE breath by breath smoothed"
USE_SMOOTHED_VEBXB = False


def plot_ve_comparison(vo2bxb_file: Path, vebxb_file: Path, participant: str, visit: str, output_dir: Path):
    vo2 = pd.read_csv(vo2bxb_file)
    vebxb = pd.read_csv(vebxb_file)

    # Check of verplichte kolommen bestaan
    required_vo2_cols = {"t", "VE"}
    if not required_vo2_cols.issubset(vo2.columns):
        raise ValueError(f"Ontbrekende kolommen in {vo2bxb_file.name}: {required_vo2_cols - set(vo2.columns)}")

    if USE_SMOOTHED_VEBXB:
        vebxb_col = "VE breath by breath smoothed"
    else:
        vebxb_col = "VE breath by breath"

    required_vebxb_cols = {"t", vebxb_col}
    if not required_vebxb_cols.issubset(vebxb.columns):
        raise ValueError(f"Ontbrekende kolommen in {vebxb_file.name}: {required_vebxb_cols - set(vebxb.columns)}")

    # Alleen relevante kolommen
    vo2 = vo2[["t", "VE"]].dropna().copy()
    vebxb = vebxb[["t", vebxb_col]].dropna().copy()
    vebxb = vebxb.rename(columns={vebxb_col: "VE_bxb"})

    # Sorteren op tijd
    vo2 = vo2.sort_values("t")
    vebxb = vebxb.sort_values("t")

    # Duplicaten op tijd verwijderen (voor veilige interpolatie)
    vo2 = vo2.drop_duplicates(subset="t", keep="first")
    vebxb = vebxb.drop_duplicates(subset="t", keep="first")

    t_vo2 = vo2["t"].to_numpy(dtype=float)
    ve_vo2 = vo2["VE"].to_numpy(dtype=float)

    t_bxb = vebxb["t"].to_numpy(dtype=float)
    ve_bxb = vebxb["VE_bxb"].to_numpy(dtype=float)

    if len(t_vo2) < 2 or len(t_bxb) < 2:
        raise ValueError("Te weinig datapunten om te plotten.")

    # Gemeenschappelijke tijd-as op overlap
    t_min = max(np.min(t_vo2), np.min(t_bxb))
    t_max = min(np.max(t_vo2), np.max(t_bxb))

    if t_max <= t_min:
        raise ValueError("Geen overlappende tijd-as tussen VO2BxB en VEBxB.")

    t_common = np.arange(np.ceil(t_min), np.floor(t_max) + 1, 1)

    if len(t_common) < 2:
        raise ValueError("Te weinig overlap voor interpolatie.")

    ve_vo2_interp = np.interp(t_common, t_vo2, ve_vo2)
    ve_bxb_interp = np.interp(t_common, t_bxb, ve_bxb)
    diff = ve_vo2_interp - ve_bxb_interp

    mean_diff = np.mean(diff)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Bovenste plot: beide VE-signalen
    axes[0].plot(t_vo2, ve_vo2, label="VE from VO2BxB")
    axes[0].plot(t_bxb, ve_bxb, label=f"VE from VEBxB{' (smoothed)' if USE_SMOOTHED_VEBXB else ''}")
    axes[0].set_ylabel("VE (L/min)")
    axes[0].set_title(f"{participant} {visit}: VE vergelijking")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Onderste plot: verschil
    axes[1].plot(t_common, diff, label="VO2BxB VE - VEBxB VE")
    axes[1].axhline(0, linestyle="--")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Verschil (L/min)")
    axes[1].set_title(
        f"Verschil na interpolatie | mean={mean_diff:.2f}, MAE={mae:.2f}, RMSE={rmse:.2f}"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{participant}_{visit}_VE_vs_VEBxB.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_file


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_files = []
    skipped = []

    participant_dirs = sorted([p for p in BASE_PREPARED_DIR.iterdir() if p.is_dir()])

    for participant_dir in participant_dirs:
        participant = participant_dir.name

        visit_dirs = sorted([v for v in participant_dir.iterdir() if v.is_dir()])

        for visit_dir in visit_dirs:
            visit = visit_dir.name

            vo2bxb_file = visit_dir / f"{participant}{visit}VO2BxB.csv"
            vebxb_file = visit_dir / f"{participant}{visit}VEBxB.csv"

            output_file = OUTPUT_DIR / f"{participant}_{visit}_VE_vs_VEBxB.png"

            # Skip als figuur al bestaat
            if output_file.exists():
                print(f"SKIP {participant}/{visit}: figuur bestaat al")
                skipped.append((participant, visit, "figure already exists"))
                continue

            if not vo2bxb_file.exists() or not vebxb_file.exists():
                skipped.append((participant, visit, "bestand ontbreekt"))
                print(f"SKIP {participant}/{visit}: VO2BxB of VEBxB file ontbreekt")
                continue

            try:
                out_file = plot_ve_comparison(
                    vo2bxb_file=vo2bxb_file,
                    vebxb_file=vebxb_file,
                    participant=participant,
                    visit=visit,
                    output_dir=OUTPUT_DIR,
                )
                saved_files.append(str(out_file))
                print(f"OK   {participant}/{visit} -> {out_file.name}")

            except Exception as e:
                skipped.append((participant, visit, str(e)))
                print(f"SKIP {participant}/{visit}: {e}")

    # Optioneel overzicht opslaan
    summary_file = OUTPUT_DIR / "VE_VEBxB_plot_summary.csv"
    summary_rows = []

    for f in saved_files:
        summary_rows.append({"status": "saved", "file": f})

    for participant, visit, reason in skipped:
        summary_rows.append({
            "status": "skipped",
            "participant": participant,
            "visit": visit,
            "reason": reason
        })

    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)

    print("\nKlaar.")
    print(f"Aantal opgeslagen figuren: {len(saved_files)}")
    print(f"Overzicht opgeslagen in: {summary_file}")


if __name__ == "__main__":
    main()