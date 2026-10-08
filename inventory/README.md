# Data inventory

What data each visit of both cohorts has, whether it loads, and what it is missing (handoff §2.8).
Standalone: it uses the existing loaders (`VO2Loader`, `NIRSLoader`, `VELoader`) read-only and changes nothing else.

## Run

From the repo root, in a venv with `pandas`, `numpy`, `matplotlib`, `openpyxl` and `xlrd`:

```
python -m inventory                                      # all visits
python -m inventory --visits L03_Ramp L12_PRBS G11_Hard  # a subset
python -m inventory --no-plots
```

Or `python main.py`, which calls `run_inventory(...)`; pick the subset or full run by commenting/uncommenting in its main block.

## Outputs (`outputs/inventory/`, gitignored)

| File | Contents |
|---|---|
| `inventory_visits.csv` | One row per visit folder: IDs, role (candidate / superseded / failed), sex, age, height, annotations, handoff-expected exclusions, phase times, ramp rate, Fitdays values, `flags` |
| `inventory_signals.csv` | One row per visit × signal: file selected, files ignored, load error, duration, sample rate, exercise coverage and gaps, rest coverage, NIRS fit-factor shares, NIRS sync |
| `inventory_presence.csv`, `presence.md` | Per visit × source (CPET VO₂, Tymewear/CPET HR, Tymewear VE/BR, NIRS, protocol WR, age/sex, SMM): present / partial / missing / not representative. Raw availability, no annotations |
| `summary.md` | Missing data per visit and the cross-checks (Gies annotations vs handoff, Lucas validity, repeated visits, cross-cohort duplicates) |
| `plots/<visit_id>.png` | Work rate, VO₂, VE, HR, BR, NIRS channels, TSI + fit factor on the CPET clock |
| `inventory.log` | Progress and full tracebacks of load failures |

No names or dates of birth are written. DOB is compared in memory for the duplicate check only.

## Definitions

- **Time base:** CPET seconds. Gies Tymewear is shifted by `VE_lag` from `docs/Gies/config.py`; NIRS is placed by its latest `<m>min<s>` event marker (earlier ones can be aborted attempts; the legacy preparer uses the first).
- **Exercise window:** nominal warm-up end (330 s submax, 360 s ramp) to the nominal protocol end (Gies 2250 s, Lucas 2190 s) or, for ramps, the last breath with cadence ≥ 60 rpm.
- **Coverage:** ~1 Hz and 10 Hz signals: share of exercise seconds with a finite sample. Breath signals: 1 − (time in breath gaps > 15 s) / window.
- **Flags** describe missing or suspect data. They are not inclusion decisions.
- **Per repetition:** PRBS visits also get `exercise_coverage_rep<n>` and `max_gap_s_rep<n>` per signal.

## Inclusion CSV (`inventory/inclusion/`, committed)

After a **full** inventory run: `python -m inventory.build_inclusion`, or `build_inclusion()` in `main.py`.

- `nirs_review.csv` (hand-edited): the dev's NIRS calls, one row per `visit_id, unit`. They decide NIRS only. An exclude always wins; an include never overrides a failed automatic check (the unit stays invalid, with a warning and a note in `reason`).
- `inclusion.csv` (generated): one row per candidate visit × unit (`visit`, or `rep1`/`rep2` in PRBS visits) × sensor group (`VO2, WR, HR, VENT, NIRS, static`), with `valid`, `reason`, `decided_by` (`rule` / `review`) and, on HR rows, `hr_source` (`tymewear` / `cpet`). Superseded and failed visits are left out. The pipeline reads only this file.

Automatic checks per unit, exercise seconds only (thresholds are constants at the top of `build_inclusion.py`):

- missing source, or a gap > 20 s on raw timestamps;
- HR, per visit: Tymewear if present with span ≥ 20 bpm, r vs VO₂ ≥ 0.5 and rest coverage ≥ 80 %; else the CPET strap under the same checks; else invalid;
- NIRS: fit factor ≥ 99 % in at least 95 % of the unit's seconds;
- WR: the protocol xml. G08 Medium2, G09 Hard2 and L04 PRBS have none but a defined ideal WR, so they are valid with a note.
