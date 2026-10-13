# Methodology as implemented (living document)

This document describes the methodology **as the code implements it**: data, preprocessing, inclusion, model, training, pilot, runs and evaluation. Keep it in step with the code; the paper's Methods section is drafted from it (`docs/paper/02_methods.md`).

| | |
|---|---|
| **Last updated** | 13 Oct 2026 |
| **Code state** | branch `Gies`, HEAD `c812b04` + uncommitted `runs.py`/`main.py`/test changes (§10) |
| **Stage** | Aim 1 runs complete (primary + training-setting sensitivity); training-settings pilot stopped without a choice; inferential statistics deferred (§12); residual analyses specified, not scripted (§11.6); Aim 2 approved and implemented (Appendix B); 2a/2b/2c not yet run |
| **Target journals** | *Journal of Sport and Health Science* (JSHS) or *Sports Medicine* (see `docs/paper/README.md` for what this means for the writing) |
| **Terms** | as defined in [`CONTEXT.md`](../CONTEXT.md) (*cohort, participant, protocol, visit, unit, sensor group, shared mask, domain, configuration, fold, run, seed, trajectory* …) |
| **Related** | Approved proposal: [`AIM1_IMPLEMENTATION_HANDOFF.md`](AIM1_IMPLEMENTATION_HANDOFF.md) · Requirement → module map: [`MAPPING.md`](MAPPING.md) |

**How to maintain this file**

- When a decision changes, edit the section that describes it, then add a line to the [change log](#17-change-log).
- Items still open are marked **[PENDING]** and listed in [§15](#15-pending-decisions). When one is decided, remove the marker in the text and move the row in §15 to the change log.
- Code references (`file:line` or `module.function`) point to the place to check when updating. Line numbers drift; function names are the stable anchor.
- Deviations from the approved proposal are collected in [§14](#14-departures-from-the-approved-aim-1-proposal). The paper must report every row there.

---

## Contents

1. [Study overview](#1-study-overview)
2. [Cohorts, protocols and sensors](#2-cohorts-protocols-and-sensors)
3. [Data preparation](#3-data-preparation)
4. [Data validity and inclusion](#4-data-validity-and-inclusion)
5. [Model inputs: domains and configurations](#5-model-inputs-domains-and-configurations)
6. [Cross-validation folds and scaling](#6-cross-validation-folds-and-scaling)
7. [Model architecture](#7-model-architecture)
8. [Training procedure](#8-training-procedure)
9. [Training-settings pilot](#9-training-settings-pilot)
10. [Aim 1 runs](#10-aim-1-runs)
11. [Evaluation and outputs](#11-evaluation-and-outputs)
    - [11.6 Residual analyses](#116-residual-analyses)
12. [Planned statistics (not implemented)](#12-planned-statistics-not-implemented)
13. [Reproducibility, provenance and compute](#13-reproducibility-provenance-and-compute)
14. [Departures from the approved Aim 1 proposal](#14-departures-from-the-approved-aim-1-proposal)
15. [Pending decisions](#15-pending-decisions)
16. [Known limitations and caveats](#16-known-limitations-and-caveats)
17. [Change log](#17-change-log)
- [Appendix A: the 46 configurations](#appendix-a-the-46-configurations)
- [Appendix B: Aim 2 design](#appendix-b-aim-2-design)

---

## 1. Study overview

Two master's-thesis datasets are re-analysed with one harmonised pipeline. Earlier results from either thesis are **not** reused; both cohorts are rebuilt from raw files.

- **Target:** second-by-second pulmonary oxygen uptake (V̇O₂, mL·min⁻¹), measured breath-by-breath by a metabolic cart.
- **Inputs:** ergometer work rate, wearable chest-strap signals (HR, V̇E, BR), NIRS muscle oxygenation (TSI or Hb-diff), anthropometrics (sex, skeletal muscle mass).
- **Model:** one causal temporal convolutional network (TCN), fixed architecture and fixed training procedure for every run.
- **Aim 1 (approved, implemented):** which input modalities, alone and combined, give the best continuous V̇O₂ prediction. 46 input configurations × leave-one-subject-out (LOSO) folds × 5 seeds, trained on the pooled dataset, with within-cohort runs as reference.
- **Aim 2 (approved and implemented 13 Oct 2026):** generalisation to an unseen intensity (2a), protocol (2b) and cohort (2c), each compared with a model that did see the condition. Three configurations carried forward from Aim 1 (§12). See [Appendix B](#appendix-b-aim-2-design).

**Data flow (code):**
raw files → `inventory/` (file choice, synchronisation, per-signal metrics) → `inventory/build_inclusion.py` → committed `inventory/inclusion/inclusion.csv` (validity verdicts)
raw files → `pipeline/prepare/` (per-visit 1 Hz table) → npz cache → join `inclusion.csv` → target mask → LOSO fold (scalers, target indices) → one packed training job per fold → standalone models → fold outputs → `evaluate(run)` → `summary/`.

Entry point: `main.py`, where runs are picked by (un)commenting calls. Run definitions are frozen dataclasses in `pipeline/runs.py`; there is no YAML or CLI.

---

## 2. Cohorts, protocols and sensors

### 2.1 Cohorts and participants

| | Cohort **Gies** (2024) | Cohort **Lucas** (2026) |
|---|---|---|
| Participants | 14: `G01`, `G03`–`G15` (P2 dropped out) | 13: `L01`–`L10`, `L12`–`L14` |
| Sex | 8 M, 6 F | 8 M, 5 F |
| Age (y) ¹ | 25.7 ± 3.4 (22.3–31.6) | 25.4 ± 2.5 (22.7–31.1) |
| Height (cm) ¹ | 180.2 ± 9.3 | 178.0 ± 9.9 |
| Body mass (kg) ² | 74.0 ± 9.9 | 69.9 ± 13.6 |
| SMM (kg) ² | 30.7 ± 5.1 (21.0–39.6) | 29.9 ± 6.1 (21.6–39.4) |
| Peak V̇O₂ ³ | 3.47 ± 0.64 L·min⁻¹; 47.0 ± 7.4 mL·kg⁻¹·min⁻¹ | 3.11 ± 0.87 L·min⁻¹; 44.1 ± 7.3 mL·kg⁻¹·min⁻¹ |
| Peak power ³ (W) | 348 ± 64 | 298 ± 87 |
| GET ³ | 2.22 ± 0.38 L·min⁻¹; 153 ± 31 W | 2.10 ± 0.47 L·min⁻¹; 142 ± 41 W |
| Ramp rate (W·min⁻¹) | 20 (1), 25 (6), 30 (6), 35 (1) | 20 (4), 25 (4), 30 (5) |
| Ramp duration ⁴ (min) | 11.5 ± 1.3 (8.9–13.2) | 10.4 ± 1.9 (8.0–13.5) |
| Ethics | NES ethics committee, request #240302 | VU Faculty of Behavioural and Movement Sciences VCWE, file VCWE-2026-113 |
| Protocols (visits) | Ramp, Easy, Medium, Hard | Ramp, PRBS, Interval, Step |
| Candidate visits | 55 (G14 Medium never recorded) | 52 |

Pooled (n = 27; 16 M, 11 F): age 25.6 ± 2.9 y, height 179.2 ± 9.5 cm, body mass 72.0 ± 11.8 kg, SMM 30.3 ± 5.5 kg, peak V̇O₂ 3.30 ± 0.77 L·min⁻¹ (45.6 ± 7.4 mL·kg⁻¹·min⁻¹), peak power 324 ± 79 W, GET 2.16 ± 0.42 L·min⁻¹ (147 ± 36 W; 66 ± 6 % of peak V̇O₂). Mean ± SD (range), computed 9 Oct 2026 (scratch computation, not part of the pipeline):

¹ CPET header of the ramp visit (`outputs/inventory/inventory_visits.csv`).
² Fitdays reading matched to the ramp visit (same file); SMM = skeletal-muscle % × weight. Body mass is the BIA-scale weight (the Gies draft's 74.2 ± 9.5 kg used a separate lab scale).
³ The per-participant ramp analysis sheets `data/raw/<cohort>/<P>/Ramp/MRT estimation <P>.xlsx` (the values that prescribed the intensities). Peak V̇O₂ = highest 30-s average of 10-s averaged data; relative peak uses the BIA-scale weight. Matches the original reports (Gies draft 3.47 L·min⁻¹, 348 W, GET 2.22 L·min⁻¹ / 153 W; Lucas thesis 297 W, GET 2.10 L·min⁻¹ / 142 W; Lucas thesis peak V̇O₂ 3.14 ± 0.83 differs slightly).
⁴ From the end of the 30-W warm-up (360 s) to the last breath with cadence ≥ 60 rpm.

Peak V̇O₂ is not recomputed with the pipeline's own 30-s rule (§11.3) because the characterisation should use the values that prescribed the protocols.

**IDs.** `participant_id` = cohort letter + original study number, zero-padded (`G08`). `visit_id` = `<participant_id>_<protocol>` (`G08_Medium`). The raw folder a visit came from is its `source_visit` (`Medium2`). Original numbers are kept so IDs map to lab records (`inventory/catalog.py`).

**Excluded data in the archives:** an unrelated 2025 dataset (P01–P10, LP01–LP10) is not a cohort of this study.

### 2.2 Protocols

All visits: 2 min seated rest on the ergometer, then a warm-up, then the exercise phase. Cadence 70–90 rpm. Intensities were prescribed from the participant's ramp test (gas exchange threshold, GET; peak). Δx % = GET + x % × (peak − GET), in power.

| Cohort | Protocol | Warm-up | Exercise phase | Nominal timeline (s): rest end / warm-up end / exercise end | Units |
|---|---|---|---|---|---|
| both | **Ramp** | 4 min at 30 W | linear ramp 20, 25, 30 or 35 W·min⁻¹ (chosen per participant) to task failure (cadence < 60 rpm) | 120 / 360 / last breath with cadence ≥ 60 rpm | 1 (`visit`) |
| Gies | **Easy** (L→M) | 3.5 min PRBS at the visit's levels | 2 × 16-min PRBS, 25 W ↔ 90 % GET | 120 / 330 / 2250 | 2 (`rep1` 330–1290, `rep2` 1290–2250) |
| Gies | **Medium** (L→H) | as Easy | 2 × 16-min PRBS, 25 W ↔ Δ30 % | 120 / 330 / 2250 | 2 |
| Gies | **Hard** (M→H) | as Easy | 2 × 16-min PRBS, GET ↔ Δ30 % | 120 / 330 / 2250 | 2 |
| Lucas | **PRBS** | 3.5 min at 80 % GET (constant) | 2 × 15.5-min PRBS, 25 W ↔ Δ40 %, 30-s bits | 120 / 330 / 2190 | 2 (`rep1` 330–1260, `rep2` 1260–2190) |
| Lucas | **Interval** | 3.5 min at 80 % GET | 10 × (2 min 25 W + 1 min Δ70 %) + 1 min 25 W | 120 / 330 / 2190 | 1 |
| Lucas | **Step** | 3.5 min at 80 % GET | 6 min 80 % GET → 6 min Δ30 % → 6 min 80 % GET → 3 min Δ60 % → 5 min 25 W → 5 min 60 % GET | 120 / 330 / 2190 | 1 |

All levels verified against the ergometer xml files (9 Oct 2026): Gies Hard low = GET (e.g. G01 179 W = GET 178.9 W; Easy high 161 W = 90 % GET). Lucas warm-up = 80 % GET in PRBS, Interval and Step (L01 139 W); PRBS high = Δ40 %; Interval high = Δ70 %; Step levels = 80 % GET, Δ30 %, 80 % GET, Δ60 %, 25 W, 60 % GET.

**How the intensities were derived** (both cohorts, same spreadsheet template `MRT estimation <P>.xlsx` and `GET Estimation Plots <P>.png`, on 10-s averaged ramp data):
- GET (as V̇O₂) by visual inspection of V̇O₂ vs V̇CO₂ (V-slope), V̇E/V̇O₂, V̇E/V̇CO₂, PetO₂, PetCO₂ and V̇E vs V̇O₂/V̇CO₂.
- Peak V̇O₂ = highest 30-s average; peak power from the ramp.
- A linear regression of V̇O₂ on time over the ramp, and the mean response time MRT = (baseline V̇O₂ of the last 60 s of 30-W cycling − intercept)/slope (mean 36 ± 19 s), map any V̇O₂ to ramp time; subtracting the MRT and applying the ramp rate gives the power.
- Δx % is defined on V̇O₂: GET + x % × (peak V̇O₂ − GET), then converted to power as above. Fractions of GET (90 %, 80 %, 60 %) are fractions of the GET power.

- Gies PRBS repetition: 960 s of 30-s units (runs 30–90 s); the warm-up is the sequence's first 210 s at the same levels. Lucas: 930 s = 31 × 30-s units (runs 30–150 s) (`inventory/catalog.py: NOMINAL_TIMELINES`; xml files).
- Visit order (CPET test dates): the ramp was the first visit for all 27 participants; the other visits followed in random order.
- **Visit spacing.** Gies: all four visits within 1–23 days (median span 8 days); 10 pairs of visits were on the same day, 2.2–7.6 h apart (G03, G04 and G05 did all four visits in 2 days, two per day; G06, G08, G13 and G14 had one same-day pair). Lucas: 9–39 days in total, ≥ 1 day between visits.
- **Laboratory conditions** (CPET header): Gies 19.5 ± 0.6 °C (19–21), 58 ± 3 % RH; Lucas 19.0 ± 1.2 °C (18–22), 50 ± 8 % RH. Tests between 09:00 and 19:30.
- **Repeated visit:** G08 Medium and G09 Hard were repeated after a partial failure; the complete visits (`Medium2`, `Hard2`) are used. **Failed visit:** G13 FailedRamp, repeated as G13 Ramp, never used (`inventory/catalog.py: SUPERSEDED_VISITS, FAILED_VISITS`).
- Phases and units come from the nominal timeline, not from the cart's phase labels (`pipeline/prepare/grid.py`).

### 2.3 Sensors

| Sensor | Device | Signals used | Native rate | Role |
|---|---|---|---|---|
| Metabolic cart | Cosmed Quark CPET | V̇O₂ (target); chest-strap HR (backup HR source only); header: sex, age, protocol name, test date | breath-by-breath | ground truth |
| Chest strap | Tymewear VitalPro | HR (1 Hz series); V̇E and BR (breath-by-breath columns, one breath per ~2.5 s) | 1 Hz / per breath | inputs HR, VENT |
| NIRS | Artinis PortaMon (Gies: original, 1 receiver, 10 Hz file; Lucas: MKIII, 100 Hz file), right vastus lateralis | TSI (%), O₂Hb and HHb of channel Tx3 | exported at 10 Hz (all 110 files) | input NIRS |
| Ergometer | Monark LC7TT | protocol definition (xml), not measured power | — | input WR |
| BIA scale | Sacoma smart scale (Fitdays app) | skeletal-muscle %, body weight, measured before each visit | per visit | input ANTHRO |

The cart's own V̇E and respiratory frequency are **never** inputs: they come from the same flow signal as V̇O₂.

---

## 3. Data preparation

Code: `inventory/` (file choice, loading, sync) and `pipeline/prepare/` (features). Output: one processed 1 Hz table per visit, cached (§13).

### 3.1 File selection

`inventory/catalog.py: _choose` keeps the files that pass a preference test and takes the first by name (recording "ambiguous" if several pass).

| Sensor | Rule |
|---|---|
| CPET | Breath-by-breath export (`BxB` in name). 10-s averaged exports are never used, even if alone (regex `(?<!\d)10[ _]?s(ec)?(?![a-z])`). Lock files `~$…` ignored. |
| NIRS | `.xlsx` export; `.oxy4/.oxy5/.txt` ignored. Duplicate columns (L04 Step `TSI%` ×5, identical): fullest kept. |
| Tymewear | `.csv` containing `VE`. |
| Protocol xml | Name contains the protocol name; the `PRBSV2 STEP` file in Lucas PRBS folders is ignored. G01 Easy has two candidates; the first is used. |
| Fitdays | Participant-level file starting with `fitdays` (actually `.xls`). |

Legacy faults this fixes: first-keyword match picked the 10-s V̇O₂ export for 7 Lucas ramps (L03, L05, L06, L07, L10, L13, L14), and lock files.

### 3.2 Synchronisation (CPET clock is the master)

- **Gies Tymewear:** time − `VE_lag`, the per-visit lag hand-annotated in Gies's original config (`docs/Gies/config.py`; 27–3844 s, the large values are clock offsets). Applied to both the 1 Hz series and the breath table.
- **Lucas Tymewear:** already on the CPET clock (lag 0).
- **NIRS:** the export's **latest** parsable `XminYY` event marker is the sync point: sample i is at `(i − marker_row)/f + event_s`, f = header "Export sample rate". The latest marker is used because 8 Gies files have earlier markers from aborted attempts (G01 Ramp: first marker would put exercise onset ~360 s late). No manual per-visit corrections. An unsynced export would make NIRS absent (none occur).
- **Work rate:** reconstructed on the CPET clock from the protocol definition (§3.4).

### 3.3 Time grid and phases

`pipeline/prepare/grid.py: build_grid`

- Integer 1 Hz grid on the CPET clock, t = 0 … exercise_end − 1.
- `phase`: rest [0, rest_end), warm-up [rest_end, warmup_end), exercise [warmup_end, exercise_end). Rest end = 120 s; warm-up end 330 s (submax) or 360 s (ramp).
- `unit`: `rep1`/`rep2` for PRBS-type visits, else `visit`; only exercise seconds belong to a unit.
- **Ramp end** = the last breath with cadence ≥ 60 rpm (`inventory/sensors.py`), i.e. task failure.
- Rest never enters an input window: the earliest context second is 330 − 187 = 143 s (submax) or 360 − 187 = 173 s (ramp).

### 3.4 Work rate (`WR`)

`pipeline/prepare/wr.py`. The feature is the **ideal protocol work rate** (W), never the measured power, which is logged per breath and inconsistent (ideal vs measured: median |Δ| 0–1 W over exercise in all 28 ramps; legacy prepared ramp WR was 10–21 W off).

- **Submax visits:** step function from the ergometer xml (`ErgoProtocolStep` Time/Load1); W(t) = load of the last step with Time ≤ t. Seconds past the xml end are not present.
- **Ramps:** from the CPET protocol name (e.g. `VO2max_25Watt`): 0 W for t < 120 s, 30 W until 360 s, then `30 + step × floor((t − 360)/interval)`. Ergometer step patterns: 20 W·min⁻¹ = 1 W/3 s; 25 = 5 W/12 s; 30 = 1 W/2 s; 35 = 7 W/12 s; L04 (`VO2Ramp20W`) = 4 W/12 s. The first increment comes one interval after 360 s (assumed; matches measured power).
- **G08 Medium2, G09 Hard2** (no xml): the superseded sibling's xml (same CPET protocol).
- **L04 PRBS** (no xml): another Lucas PRBS xml's timing; rest 0 W; low 25 W; high level and warm-up level = median measured power over those seconds of L04's own CPET file (high 120 W, warm-up 79 W; warm-up median skips 15 s at each edge).
- Legacy fault fixed: both legacy preparers started the ramp 30 s early.

### 3.5 Breath-by-breath cleaning (V̇O₂, V̇E, BR)

`pipeline/prepare/breath.py` — exact, vectorised port of the legacy helpers; identical for both cohorts (tested against legacy output).

1. Keep breaths with finite time and value; sort by time.
2. **Outlier rejection:** breath i is removed if |xᵢ − mean(xᵢ₋₅…xᵢ₋₁)| ≥ 3 × SD(xᵢ₋₅…xᵢ₋₁) (sample SD, raw previous 5 breaths; the first 5 breaths are kept).
3. **5-breath median filter:** centred for V̇O₂ (target), trailing for V̇E and BR (inputs, causal). A window containing a removed breath gives NaN.
4. **Linear interpolation** to the integer 1 Hz grid between the first and last finite filtered breath.
5. **Presence:** second t is *present* only if the two raw breaths around it are ≤ 20 s apart (`max_interp_gap_s = 20`) and the interpolated value is finite. Interpolated values across longer gaps exist but are flagged not present.

- V̇O₂: CPET breaths. V̇E, BR: raw Tymewear `VE breath by breath` / `BR breath by breath` columns (never Tymewear's smoothed columns). VENT is present only where both V̇E and BR are present.
- Side effect (as in legacy): when one of the last breaths is rejected, the trailing median is NaN for the remaining breaths, so VENT is absent in the last 1–11 s of exercise in 10 Lucas submax visits; those final targets drop out of the shared mask.

### 3.6 Heart rate and heart-rate reserve (`HR`, `HRR`)

`pipeline/prepare/hr.py`

- **Source** per visit, from `inclusion.csv` (§4.2): Tymewear HR if it passes the HR check, else the CPET chest strap (G06 Ramp, G06 Easy, G09 Ramp, G11 Easy, G13 Medium), else none (HR absent).
- HR: no smoothing; linear interpolation to 1 Hz; present where samples are ≤ 20 s apart.
- **HRR = (HR − HRrest) / (HRpeak − HRrest)**
  - HRrest = mean 1 Hz HR over present seconds in [0, 120 s) of the **same visit and source**.
  - HRpeak = **208 − 0.7 × age** (Tanaka) for all participants; age = the visit's CPET header age at full precision.
  - Override: L14 Interval takes its age from L14 PRBS (its CPET header carries another participant's profile; §3.8).
- Legacy (both theses) used min/max HR across visits, baked in at preparation.

### 3.7 NIRS (`TSI`, `HBDIFF`)

`pipeline/prepare/nirs.py`

- **TSI:** raw TSI (%), not normalised.
- **Hb-diff:** O₂Hb − HHb of channel **Tx3**, raw as exported: no occlusion or within-visit normalisation, no baseline correction (`hbdiff_channel = tx3`, `hbdiff_baseline = none`). Tx3 is intended to be the 40-mm source–detector distance (Gies's original choice); **this mapping is unverified** on both devices.
- Row cleaning: drop all-zero rows (Gies exports often end with one) and a nonphysical first row (|ΔHb| > 150 µM).
- **10 Hz → 1 Hz:** second t = mean of finite samples in [t, t+1). No interpolation; NIRS present where both TSI and Hb-diff are finite (no gap tolerance).
- Not used: occlusion calibration, fit factor as a feature, Lucas's native `HbDiff` column, the MKIII short-separation channel (Gies has neither).
- **Zero point:** no source states when OxySoft zeroes ΔHb; raw Hb-diff carries an arbitrary per-visit (even per-export) offset. Rest means range −5.5 to +7.8 µM against exercise means of −1.8 to −44 µM in 16 checked visits (`outputs/hbdiff_baseline/offsets.csv`). Rest-mean subtraction was judged visually to make little difference, so raw is the default. Flag `hbdiff_baseline = rest_mean` exists.

### 3.8 Anthropometrics (`sex`, `SMM`)

`pipeline/prepare/anthro.py`, `pipeline/prepare/overrides.py`

- **Sex** from the CPET header: male = 1, female = 0; never scaled.
- **SMM (kg)** = Fitdays skeletal-muscle % × body weight, from the reading matched to the visit by date (CPET test date); same-day readings are disambiguated by the Fitdays `Visit` label. Matching fails loudly. Per visit (`smm = per_visit`).
- Note: the legacy code and Lucas's thesis used Fitdays "Muscle mass", a different, larger quantity.
- Age is HRpeak metadata only, never an input.
- **Overrides** (`inventory/inclusion/prep_overrides.csv`, part of the cache key):
  - L14 PRBS: no Fitdays reading on the test day (15/04/2026) → nearest earlier reading (09/04, 08:28).
  - L14 Interval: two readings on the day (0.09 kg apart) → the first (10:11).
  - L14 Interval: age from L14 PRBS. The wrong participant profile (L13) was selected in the CPET software for this visit; the recorded data are L14's (confirmed with Lucas, 9 Oct 2026). Sex is the same for both, and height is not used, so the age override is the only correction needed.

### 3.9 Processed visit table

One row per grid second: `t, phase, unit`, features `WR, HR, HRR, VE, BR, TSI, HBDIFF, sex, SMM`, target `VO2`, and a `present_<group>` flag per sensor group. Built by `pipeline/prepare/visit.py`, cached by `pipeline/cache.py`.

---

## 4. Data validity and inclusion

Validity is decided **once**, outside the training code, and committed as `inventory/inclusion/inclusion.csv`. The pipeline only reads it (`pipeline/inclusion.py`).

### 4.1 Units and sensor groups

- **Unit:** a PRBS repetition (Gies Easy/Medium/Hard, Lucas PRBS), otherwise the visit's whole exercise phase. 161 units (Gies 96, Lucas 65).
- **Sensor groups:** `VO2`, `WR`, `HR`, `VENT`, `NIRS`, `static` (sex + SMM).
- A group is valid or invalid for a **whole unit**, never per second: a bad second would also corrupt the next 187 s of input windows.
- Checks look at the unit's exercise seconds only; warm-up quality never invalidates a unit.
- `inclusion.csv`: 966 rows = 161 units × 6 groups, with `valid`, `reason`, `decided_by` (`auto`/`review`), `hr_source`.

### 4.2 Layer 1: automatic checks

`inventory/build_inclusion.py` (thresholds at the top of the file), computed from the inventory's per-signal metrics (`inventory/metrics.py`).

| Group | Rule |
|---|---|
| all signals | **Missing** if no usable file or no exercise coverage. **Gap rule:** invalid if the longest gap in the unit's exercise window exceeds 20 s (breath signals: inter-breath gap incl. window edges; ≈1 Hz series: longest run of empty seconds). |
| VO₂ | gap rule on CPET V̇O₂ |
| WR | valid if the protocol xml / ramp name loaded; named exceptions G08 Medium, G09 Hard (sibling xml), L04 PRBS (rebuilt) |
| HR | per visit, sources in order Tymewear then CPET; a source passes if present, **span (p95 − p5) ≥ 20 bpm** and **Pearson r vs 30-s smoothed V̇O₂ ≥ 0.5**, both over 0 s → exercise end, and **rest coverage ≥ 80 %** of 0–120 s. First passing source is the visit's HR source; each unit then also needs the gap rule on that source. |
| VENT | gap rule on Tymewear V̇E and BR |
| NIRS | gap rule on TSI and Hb-diff, **and** ≥ 95 % of the unit's NIRS samples with **fit factor ≥ 99 %** |
| static | sex and Fitdays skeletal-muscle % available |

Rationale for the less obvious thresholds:
- **HR check:** coverage alone misses stuck sensors (e.g. G07 Ramp CPET HR ~118–120 bpm throughout, span 5 bpm, r ≈ 0). The window starts at rest because during Hard PRBS exercise genuine HR barely varies (span < 20 bpm, r ≈ 0.1) and the exercise-only version gave false failures. The final thresholds reproduce Gies's original `HR_source` annotations exactly.
- **Fit factor 99 % / 95 %:** the fit factor is Artinis's agreement index across the three source–detector distances for TSI; no published threshold exists. Lucas's 99.9 % was a placement check at acquisition; applied per sample it would drop 9 Lucas visits entirely. 99 % removes nothing in Lucas (per-sample range 99.47–100 %) and matches Gies's draft (> 99 %). On current data the rule invalidates no candidate unit.
- **20-s gap:** legacy interpolated across gaps of any length. The value 20 s is a judgement call.

### 4.3 Layer 2: NIRS review

The developer's visual review (`inventory/inclusion/nirs_review.csv`, 12 rows, NIRS group only). A review **exclude always wins**; a review **include never overrides a failed automatic check**. The original Gies annotations are retired for validity.

- Exclude: G08 Easy rep 1 + 2 (unusable), G07 Hard rep 2 (unusable), G07 Medium rep 1 (warm-up TSI artefact at ~150–280 s enters the input windows of rep 1's first targets).
- Include (where annotations had excluded): G07 Ramp, G07 Medium rep 2, G11 Easy, G11 Hard rep 1 + 2, G11 Ramp, G15 Ramp (whole ramp; fit factor drops near the end but the unit passes the rule).

### 4.4 Result

| | Visits included (shared mask) | Units valid |
|---|---|---|
| Gies | **47 of 55** (G04 Hard rep 1 only; G07 Medium rep 2 only) | 83 of 96 |
| Lucas | **52 of 52** | 65 of 65 |
| Total | **99 of 107** | 148 of 161 |

Fully excluded Gies visits and the reason:

| Visit | Failing group(s) |
|---|---|
| G06 Ramp | VENT (no Tymewear file) |
| G07 Ramp | HR (Tymewear missing, CPET HR stuck), VENT (no Tymewear) |
| G09 Ramp | VENT (no Tymewear file) |
| G11 Ramp | HR (Tymewear missing, CPET HR stuck), VENT (no Tymewear) |
| G13 Ramp | HR (Tymewear HR empty, CPET HR stuck) |
| G13 Easy | HR (Tymewear HR empty, CPET HR stuck) |
| G07 Hard | HR (Tymewear r = 0.00, no CPET HR); NIRS rep 2 (review) |
| G08 Easy | NIRS (review) |

Partial: G04 Hard rep 2 (cart recording stops at ~1355 s; V̇O₂ gap 894 s), G07 Medium rep 1 (review).

HR source over 107 visits: 97 Tymewear, 5 CPET strap, 5 none.

Differences from the proposal's expected exclusions (≈ 44 of 56): see `MAPPING.md` §8. **Target seconds (shared mask):** 157,382 s (43.7 h) of 168,976 exercise seconds: Gies 76,933 (Ramp 5,925; Easy 23,030; Medium 23,979; Hard 23,999), Lucas 80,449 (Ramp 8,061; PRBS 24,095; Interval 24,146; Step 24,147). The presence rule removes only 205 s from valid units (31 units, 1–32 s each, at recording ends). Source: `outputs/runs/smoke/summary/targets.csv` (the smoke run applies the same shared mask to every cached visit); `folds.csv` agrees (G07 fold: 154,510 train + 2,872 test). Re-check against `A1_pooled/summary/targets.csv`.

### 4.5 Target definition and masks

`pipeline/inclusion.py`. Second t of a visit is a **target** iff:
1. t is in the exercise phase;
2. t's unit is valid for every required sensor group;
3. V̇O₂ is present at t;
4. every required input group is **present** on all of t − 187 … t − 1 within the same visit.

Context seconds need only be present, not in a valid unit: the warm-up end and an invalid neighbouring repetition can serve as context (G07 Medium: first target t = 1290 s, context 1103–1289 s from rep 1).

- **Shared mask (primary):** required groups = all six. Every configuration in a fold trains and is tested on identical seconds.
- **`per_config` mask (sensitivity):** required groups = VO₂ + the configuration's own groups (`static` only with ANTHRO). Configurations with bitwise-identical target sets share a training job. With `per_config`, `WR` alone uses all 55 Gies visits.

---

## 5. Model inputs: domains and configurations

`pipeline/domains.py`

| Domain | Token(s) | Features | Sensor | Scaling |
|---|---|---|---|---|
| WR | `WR` | work rate | ergometer protocol | min–max |
| HR | `HR` | HR, HRR | Tymewear (CPET strap backup) | z-score each |
| VENT | `VENT` | V̇E, BR | Tymewear | z-score each |
| NIRS | `TSI` **or** `HBDIFF` | TSI or Hb-diff | PortaMon | z-score |
| ANTHRO | `ANTHRO` | sex, SMM | CPET header, BIA scale | sex none; SMM z-score |

- A domain's features always enter together (HR always with HRR); NIRS enters through exactly one NIRS variable.
- **Configuration** = non-empty set of time-varying domains, NIRS as at most one variable, ± ANTHRO. ANTHRO never alone.
- Count: (2³ × 3 − 1) time-varying sets × 2 = **46** (`nirs_exclusive = True`). `False` gives the proposal's 62.
- IDs join tokens in canonical order WR, HR, VENT, TSI, HBDIFF, ANTHRO (`WR+HR+TSI+ANTHRO`); generated programmatically, ordered by size then canonical order. Full list in [Appendix A](#appendix-a-the-46-configurations).
- Input dimension: 1 (e.g. `WR`) to 8 (`WR+HR+VENT+TSI+ANTHRO`).

---

## 6. Cross-validation folds and scaling

`pipeline/folds.py`

- **LOSO** in every run: one fold per participant of the run's cohorts; the training set is every included visit of all other participants. Pooled: 27 folds, 26 training participants; Gies: 14 folds, 13; Lucas: 13 folds, 12.
- No validation set; all non-test participants train.
- Runtime checks (explicit exceptions): test participant ∉ training set; unique test IDs; scaler-fit rows contain no test-participant row; last gathered window row = t − 1; all inputs and targets finite.
- **Scalers fit per fold on training participants only:**
  - V̇O₂ (target): z-score on the training **target** seconds.
  - Features: on the rows read by at least one training target's window (includes warm-up and context seconds; never crosses a visit; never test data). WR min–max; other continuous features z-score (population SD); sex unscaled.
  - Applied to the test participant unchanged; predictions inverse-transformed to mL·min⁻¹ before any metric.
- Aim 2 hook: `Fold(test_participant, train_visits)` accepts an explicit training-visit list. The subsampling of the Aim 2 seen arm (Appendix B.4) is `aim2.subsampled_mask`: a per-fold copy of the target mask with the dropped training targets cleared, passed to the unchanged `train.train_fold`.

---

## 7. Model architecture

`pipeline/TCN.py` — PyTorch port of the legacy Keras TCN with Keras defaults pinned.

- Input: window of 187 s × F features (t − 187 … t − 1). Output: scaled V̇O₂ at t (single value).
- **Two residual blocks:**
  - Block 1: three causal dilated Conv1D layers, dilations 1, 2, 4.
  - Block 2: two causal dilated Conv1D layers, dilations 8, 16.
  - Every conv: kernel 7, 16 filters, left (causal) zero padding (k − 1)·d.
  - After every conv: layer normalisation over channels (ε = 1e-3) → ReLU → dropout 0.2 (training only, inverted).
  - Residual: block output = conv path + 1×1 convolution of the block input (always present, also 16 → 16). No activation after the sum.
- **Head:** dense layer 16 → 1 on the last time step.
- **Receptive field:** 1 + (7 − 1) × (1 + 2 + 4 + 8 + 16) = **187 s** (checked against the window length at import).
- **Initialisation:** Glorot-uniform kernels (fan-in c_in·k, fan-out c_out·k), zero biases, LayerNorm γ = 1, β = 0; drawn from a CPU generator so weights are device-independent.
- ≈ 8–9 k trainable parameters (7.9 k at F = 1, 8.7 k at F = 8).
- Implementation note: every 1×1 projection is computed as a matrix product (`pointwise_conv`) because the Intel B580's `Conv1d` backward returned wrong gradients for a 1×1 convolution with one input channel; same function on every device.
- Earlier descriptions ("5 causal dilated Conv1D layers, residual blocks") describe the same network: 5 dilated convolutions grouped into 2 residual blocks.

---

## 8. Training procedure

`pipeline/fit.py`, `pipeline/train.py`, `pipeline/runs.py`. **One procedure for every run** (pooled, within-cohort, later Aim 2); only the training data differ.

| Setting | Value |
|---|---|
| Optimiser | Adam, β = (0.9, 0.999), ε = 1e-7 (Keras), no weight decay, constant learning rate |
| Learning rate × epochs | **5e-5 × 30** (primary; `runs.A1_TRAINING`, the proposal's default). Sensitivity: 5e-4 × 1 (`runs.A1_SHORT_TRAINING`, §10). No pilot choice (§9). |
| Batch size | 32, last partial batch kept |
| Shuffle | on: a new seeded permutation of the training targets every epoch |
| Loss | MSE on scaled V̇O₂ |
| Validation set / early stopping | none; all epochs run |
| Checkpoint | the epoch with the lowest epoch-average **training** loss (batch-size weighted, dropout active, Keras-style), strict `<` (earliest wins ties). Never chosen on test data. |
| Seeds | 5 (0–4) per configuration × fold; all saved; **no seed selection** |
| Determinism | `torch.use_deterministic_algorithms(True)`; same job on same device is bit-identical |

**Seeds and random streams** (`provenance.derive_seed` = SHA-256 of the parts):
- Initial weights: from (seed, configuration ID).
- Shuffle order: from seed only; one permutation per epoch, shared by all configurations with that seed.
- Dropout: one generator per packed job, from (run, test participant, mask group).
- No global seeding of Python/NumPy/torch; all randomness goes through explicit generators.

**Packed training** (efficiency; no effect on the function learned):
- One job per fold (per identical-mask group under `per_config`) trains all 46 configurations × 5 seeds = K = 230 models at once as grouped convolutions with per-model LayerNorm; each model reads only its own feature columns (channel masks; Glorot uses the true fan-in).
- Loss = sum of per-model MSEs; parameters are disjoint, so each model gets exactly its own gradient.
- Training computes only the last output position (strided "last-only" forward; same function and gradients, parity-tested).
- After training, each model is extracted into a standalone TCN; **all predictions use the standalone models** (full forward, eval mode).
- Windows are gathered on the device from target indices.

**Per-fold outputs** (`outputs/runs/<run_id>/<participant_id>/`, written to `.tmp` and renamed when complete; finished folds are skipped on rerun):
- `predictions.parquet`: `run_id, config_id, seed, fold, participant_id, cohort, protocol, visit_id, t_s, work_rate, vo2_true, vo2_pred` (mL·min⁻¹, test participant's target seconds).
- `visit_metrics.csv`: RMSE, MAE, bias, n seconds per configuration × seed × visit.
- `epochs.csv`: training loss per epoch and model; `is_best`.
- `models.pt`: all standalone state dicts + checkpoint epochs.
- `manifest.json`: see §13.

---

## 9. Training-settings pilot

`pipeline/pilot.py`, `runs.PILOT`. Purpose: choose **one** global learning rate × epochs setting, then fix it for all runs; also measure time per training.

| Arm | Learning rate | Epochs | Read from trajectory |
|---|---|---|---|
| R (reference, Lucas's thesis) | 5e-5 | 30 | 5e-5 × 30 |
| A | 5e-5 | 10 | 5e-5 × 30 (first 10 epochs) |
| B | 2e-4 | 10 | 2e-4 × 10 |
| C (Gies's original) | 5e-4 | 1 | 5e-4 × 3 (first epoch) |
| D | 5e-4 | 3 | 5e-4 × 3 |

- Only three trajectories are trained; with a constant learning rate, an arm with fewer epochs is the prefix of a longer trajectory. A truncated arm is scored at e* = argmin over e ≤ E of the training loss (the same checkpoint rule as §8).
- **Splits: LOSO** in three families: pooled (27 splits, train on 26), Gies (14, train on 13), Lucas (13, train on 12). 54 splits × 3 trajectories = 162 jobs, ≈ 27 h.
- **Configurations (5):** `WR`, `WR+HR+VENT+TSI+ANTHRO`, `WR+HR+VENT+HBDIFF+ANTHRO`, `HR+VENT+TSI+ANTHRO`, `HR+VENT+HBDIFF+ANTHRO`. **Seeds:** 0, 1. Shared mask.
- After every epoch, each model is scored on the held-out participant (RMSE in mL·min⁻¹ per visit).
- **Aggregation** (`decide`): visit RMSE → participant mean → mean over splits × seeds, per cell; 15 cells = 3 families × 5 configurations.
- **Rule (advisory, never applied):** an arm qualifies in a cell if its RMSE ≤ the cell's best + 10 mL·min⁻¹; choose the fewest-epoch arm that qualifies in all 15 cells (A/B tie → lower mean RMSE); else R. Output would have been `outputs/pilot/decision.md`.

**Outcome (dev, 9 Oct 2026): stopped after 24 of 162 jobs, no setting chosen.**
- Completed: pooled splits G01, G03–G09 (8 splits × 3 trajectories; all held-out participants from cohort G). Logs kept in `outputs/pilot/`; `decide()` was not run.
- Partial results (Claude's recomputation of `decide()`'s scoring over the 8 splits, 5 configurations × 2 seeds): arm means C 260, B 264, D 268, A 268, R 269 mL·min⁻¹, while the mean |seed 0 − seed 1| difference was 35 mL·min⁻¹. No arm met the 10 mL·min⁻¹ rule in every configuration. Configurations with WR improved slightly with more epochs (WR: C 194 → R 183); configurations without WR got worse (HR+VENT+TSI+ANTHRO: 328 → 371).
- Time per pooled split at K = 10: R 29 min, B 9.4 min, D 2.9 min (41 min for all three); epoch time hardly depends on K (≈ 55 s at K = 10, ≈ 57 s at K = 2).
- **Why stopped:** arms within seed noise, and the remaining pilot (≈ 23 h) cost about as much as Aim 1 at R. **Replacement:** every A1 run is trained at R (primary, fixed before any A1 result) and at C (sensitivity), §10.
- **Consequence to report:** C was chosen as the sensitivity setting after seeing partial pilot results whose held-out participants are Aim 1 test participants (see §14). The primary setting is the proposal's default and uses no test-fold information.

---

## 10. Aim 1 runs

`pipeline/runs.py`. All: 46 configurations × 5 seeds × LOSO, default `PrepSettings`. Primary runs use `A1_TRAINING` (5e-5 × 30); the training-setting sensitivity runs are identical except for `A1_SHORT_TRAINING` (5e-4 × 1) and their run IDs.

| Run ID | Cohorts | Folds | Mask | Training | Trainings | Status |
|---|---|---|---|---|---|---|
| `A1_pooled` | Gies + Lucas | 27 | shared | 5e-5 × 30 | 6,210 | done 10 Oct |
| `A1_gies` | Gies | 14 | shared | 5e-5 × 30 | 3,220 | done 10 Oct |
| `A1_lucas` | Lucas | 13 | shared | 5e-5 × 30 | 2,990 | done 11 Oct |
| `A1_{pooled,gies,lucas}_lr5e-4_ep1` | as above | 27/14/13 | shared | 5e-4 × 1 | as above | sensitivity, done 9 Oct |
| `A1_*_per_config` | as above | 27/14/13 | per_config | 5e-5 × 30 | as above | sensitivity, optional, not run |
| `smoke` | both | G07, L04 | shared | 5e-5 × 3 | 2 configs × 1 seed | plumbing check only, done |

Total primary: 12,420 trainings in 54 packed jobs (and the same again for the sensitivity setting). Measured wall time on the B580: primary ≈ 32 h for all three runs (≈ 47 min per pooled fold, ≈ 22 min per within-cohort fold); sensitivity ≈ 1.5 h. Training-set sizes (participants, target seconds) per fold are reported in `summary/folds.csv`.

---

## 11. Evaluation and outputs

`pipeline/metrics.py`, `pipeline/evaluate.py`. All metrics in float64 on mL·min⁻¹, computed on the test participant's target seconds only. `evaluate(run)` refuses to run if a fold is missing or if any fold's training settings differ from the run's.

### 11.1 Primary metric: RMSE (also MAE, bias)

Per visit (all target seconds of the visit; a PRBS visit covers its valid repetitions): error = predicted − measured; RMSE = √mean(error²); MAE = mean|error|; bias = mean(error).

Aggregation, per configuration:
1. seconds → **visit** metric (per seed);
2. visit → **participant**: mean of the participant's visit metrics (per seed);
3. participant → **seed-averaged** participant value: mean over the 5 seeds (the *seed-averaged metric*; never the metric of the averaged prediction);
4. reported: **mean ± SD across participants** of the seed-averaged values, and **SD across seeds** of the per-seed configuration means.

### 11.2 Secondary metrics

Per configuration × seed × group, over all target seconds of the group; reported as mean and SD across seeds:
- **Pearson r** between measured and predicted V̇O₂.
- **Repeated-measures correlation** r_rm (`pingouin.rm_corr`, subject = participant).
- **Repeated-measures Bland–Altman** (Bland & Altman 2007, "true value varies"): d = predicted − measured; bias = mean(d) over all seconds; one-way ANOVA of d on participant gives MS_between, MS_within; m̃ = (N² − Σmᵢ²)/((n − 1)N); σ² = max(0, (MS_b − MS_w)/m̃) + MS_w; 95 % LoA = bias ± 1.96σ.
- No p-values for correlations (~150 k autocorrelated seconds).

### 11.3 VO₂peak (ramps only)

Highest 30-s trailing rolling mean over the ramp visit's target seconds (full windows only), for measured and each seed's predicted V̇O₂. Per visit: diff = predicted − measured peak and |diff|, per seed, then averaged over seeds. Called *VO₂peak* because no plateau criterion is checked.

### 11.4 Breakdowns

Every run: groups `all`, per cohort, per protocol (Ramp, Easy, Medium, Hard, PRBS, Interval, Step).

### 11.5 Summary files (`outputs/runs/<run_id>/summary/`)

- `summary.csv`: per configuration × group: n participants/visits/seeds; RMSE, MAE, bias (mean, SD participants, SD seeds); r_rm, r_p, BA bias and LoA (mean, SD seeds); VO₂peak diff and |diff| (mean, SD participants).
- `visit_metrics_seedavg.csv`: per configuration × visit, seed-averaged RMSE/MAE/bias (input for the statistics in §12).
- `predictions_seedavg.parquet`: per-second mean prediction over seeds (for plots and residual analyses; not used for reported metrics).
- `vo2peak.csv`, `folds.csv` (training-set sizes, wall time), `targets.csv` (exercise and target seconds per visit × unit).
- No "winner" is selected inside the pipeline.

### 11.6 Residual analyses

**Status:** specified 13 Oct 2026 (dev); scripted in `pipeline/residuals.py` → `outputs/residuals/<run_id>/`; run on `A1_pooled` 13 Oct 2026 (event counts match). Residual = predicted − measured V̇O₂ (mL·min⁻¹) throughout, as in §11.1.

**Data and scope**
- **Run:** `A1_pooled`, seed-averaged predictions (`summary/predictions_seedavg.parquet`, §11.5).
- **Configurations:** the three carried forward (§12): `WR`, `HR+VENT+TSI+ANTHRO`, `WR+VENT`. All 7 protocols.
- **Not a verbatim replication** of Lucas's runs tests (`docs/Lucas_Thesis.tex:457`): his `WR+HBDIFF+ANTHRO` on a Lucas-only run does not carry over given the Aim 1 results (dev). The method is similar; run and configurations are this study's.
- **Statistics:** descriptives only (§12). Means with a 95 % t-CI, participant as the unit. No p-values and no Holm correction, except the z and p of the runs-test table (output 1).
- **No amplitude normalisation** of residuals. ΔV̇O₂/ΔWR scaling is a plotting option for the error-by-phase figure only (undecided).
- **Contiguity (assumed):** each visit's target seconds form one contiguous block. `targets.csv` shows losses of 1–32 s per unit, all at recording ends (§4.4). The script stops with an error if a visit's target seconds are not contiguous.

**Phase labels** come from the ideal work rate the model received as input: the `work_rate` column of the predictions (`pipeline/train.py: PREDICTION_COLUMNS`, the §3.4 step function), not from a second parse of the xml (dev). The ramp phases also use `inventory/reference/get_power.csv` and task failure (§3.3).
- **Where the WR changes are found** (dev, 13 Oct): in the WR column of the run's own processed cache (§3.9; loaded by the `cache_key` in the fold manifests, never rebuilt). It is the same §3.4 series, but it also holds the rest and warm-up seconds, which the predictions lack: the predictions hold target seconds only, so the 330-s onset change and the Step's first-block clock are invisible there. The script stops with an error unless the cache WR equals the predictions' `work_rate` on every target second.
- **Events** are WR changes at exercise-phase seconds. The phase-5/6 clock counts every change, including rest → warm-up at 120 s. Phase 6 = sustained Step seconds with WR > GET (only the Δ30 % block qualifies).
- **Task failure** = the visit's last grid second + 1 (§3.3), not its last target second; they differ by up to 6 s (L06).

**`inventory/reference/get_power.csv`** (27 rows; read from the ergometer xml files on 13 Oct 2026, not generated by pipeline code). Columns `participant_id, cohort, get_w, ramp_rate_w_per_min, ramp_get_time_s, ramp_included, source`.

| | Gies | Lucas |
|---|---|---|
| `get_w` source | lower level of the Hard xml (Hard low = GET, §2.2) | xml warm-up load (step at 120 s) ÷ 0.8 (warm-up = 80 % GET); Step, Interval and PRBS agree for every participant (L04: no PRBS xml) |
| Mean vs §2.1 | 153.4 W (153 W) | 141.8 W (142 W) |

- Every participant has an xml, so the `MRT estimation <P>.xlsx` fallback is not used.
- **`ramp_get_time_s`:** the first CPET-clock second at which the §3.4 ramp WR reaches GET, `360 + ceil((GET − 30)/step) × interval` (step and interval from §3.4; L04 4 W/12 s). Crossings fall at 22–54 % of the ramp duration; L02 is lowest (22 %, GET 73.75 W, the value its protocols were prescribed from).
- **`ramp_included`:** true for the 22 ramps in the shared mask (§4.4); false for G06, G07, G09, G11, G13.
- **No MRT shift.** The column is the second the **work rate** crosses GET. The GET power was derived after subtracting the MRT (§2.2), so V̇O₂ crosses GET roughly one MRT later (36 ± 19 s); per-participant MRTs exist only in the xlsx files. Phase 7 is centred on the work-rate crossing (below).
- **Precision:** xml loads are rounded to ±1 W, which moves a crossing by at most one ramp step (2–12 s).

**Phase windows.** All windows are disjoint; target seconds outside every window form the "other" bucket.

| # | Phase | Protocols | Window | Events |
|---|---|---|---|---|
| 1 | PRBS switch up | Easy, Medium, Hard, PRBS | **0–30 s** after the switch (primary). Sensitivity: the whole run up to the next switch (both evaluated) | 968 |
| 2 | PRBS switch down | as 1 | as 1 | 930 |
| 3 | Increase | Interval, Step | 0–180 s after the change, capped at the next change | 169 |
| 4 | Decrease | Interval, Step | as 3 | 169 |
| 5 | Sustained below GET | Step | more than 180 s since the last WR change. The clock runs across the warm-up boundary, so the first Step block (330–690 s) is entirely phase 5 | 52 blocks |
| 6 | Sustained above GET | Step | the Δ30 % block, 180–360 s after the increase | 13 |
| 7 | Ramp through GET | Ramp | `ramp_get_time_s` ± 120 s, no MRT shift | 22 |
| 8 | Ramp end | Ramp | the last 120 s before task failure | 22 |

- **Check:** the event counts must match the figure drafts (column Events). They count the events in the exercise phase of every visit with target seconds, whether or not the event's window holds any (the draft's program × included visits); `events.csv` also gives the count with target seconds.
- **Sensitivity windows (phases 1–2):** each exercise-phase switch to the next change or the exercise end; PRBS seconds before the first exercise-phase switch stay "other".
- **Interval bouts** merge with the Step changes in phases 3 and 4. Interval increases last 60 s and decreases 120 s, so the cap at the next change truncates them; the figure marks 60 s.
- **The 330-s exercise-onset change** in Lucas PRBS and Interval (warm-up 80 % GET → 25 W) is **included** and labelled by direction. Its final treatment is decided later (dev).
- **Phase 7 placement:** ±120 s without a shift is the only option checked that stays clear of both the warm-up and phase 8 for all 22 ramps (±180 s fails for 6 ramps; ±120 s with a +36 s shift fails for L07, L09, L10).

**Outputs** (per configuration)
1. **Runs test joined across participants** (Wald–Wolfowitz), per protocol × configuration: n, runs, expected runs E = 2n₊n₋/n + 1 from the pooled signs, z, p and mean run length.
   - Runs are counted within visits and summed.
   - Zero residuals are dropped before counting (reported as `n_zero`).
   - At 1 Hz it is significant by construction, so it does not carry the claim. It is still run, for the supplement or as a candidate for the error-by-phase figure (dev).
2. **Per-visit run length** (carries the claim): target seconds ÷ runs, and the share of seconds in runs longer than 60 s. Summarised per protocol × configuration as the mean ± SD over participants.
3. **Per phase × configuration:**
   - **bias:** each participant's mean residual over all of their window seconds, then the mean across participants with a 95 % t-CI;
   - **error share ÷ time share:** on squared error, with "other" included so the shares sum to 1; pooled over the group's target seconds (dev, 13 Oct), so the ratio = the phase's MSE ÷ the group's MSE;
   - **rows:** groups pooled, Gies and Lucas, each with every phase it holds (shares sum to 1 within a group); the Gies and Lucas rows of interest are phases 1, 2, 7 and 8;
   - both labellings: primary and the phase-1/2 sensitivity windows;
   - not computed: the run count within phase windows (almost no runs fit inside a 30-s window).
4. **Aligned traces per phase × configuration:** measured, predicted and residual per second offset from the phase anchor, averaged per participant first, then mean ± SD across participants, with n per offset. Input for the error-by-phase figure decision.
   - **Anchors:** phases 1–4 the WR change; 5–6 the last WR change (offset = seconds since it, ≥ 180); 7 `ramp_get_time_s`; 8 task failure (offsets −120 … −1).
   - Pooled rows only; V̇O₂ in absolute values (ΔV̇O₂ is a plotting choice).

Reuse for Aim 2: not yet specified (B.9).

---

## 12. Planned statistics (not implemented)

From the approved proposal. **[PENDING, deferred]** all three items: deferred beyond the current effort (dev, 13 Oct 2026; P2), not dropped. Until then the paper reports descriptives only: means with a 95 % t-CI, participant as the unit (§11, §11.6, B.6). When resumed, they are scripted on `visit_metrics_seedavg.csv`.

- Linear mixed model: `log(RMSE) ~ configuration + protocol + cohort + (1 | participant)` (visit-level, seed-averaged RMSE). Random-effects structure to be justified (a co-author asked about random slopes on the Gies draft).
- Contrasts of each configuration against `WR` alone, Holm-corrected.
- Equivalence tests (TOST) against `WR`, margin **±25 mL·min⁻¹**, fixed before results are seen.

**Carry-forward rule for Aim 2 (decided 13 Oct 2026).** Three configurations go into every Aim 2 analysis (2a, 2b, 2c):
- `WR` alone, always, as the reference.
- Two strata, each ranked on `rmse_mean` in `A1_pooled`, group `all` (27 participants; mean over seeds and participants). The lowest in each stratum is carried forward:
  - (a) configurations without `WR` (22) → `HR+VENT+TSI+ANTHRO` (361.9 mL·min⁻¹);
  - (b) `WR` plus at least one of `HR`, `VENT` or NIRS, with `ANTHRO` optional (`WR+ANTHRO` is excluded because it adds no physiology) → `WR+VENT` (196.2 mL·min⁻¹).
- There is no simplicity or margin tie-break. The earlier "simpler configuration within the equivalence margin" idea was never an adopted rule.
- Robustness, not part of the rule: `WR+VENT` also wins on the mean of the within-cohort runs (203.0) and on the pooled model's equal-weighted cohort scores (196.3). On the within-cohort mean, the non-WR pick would be `HR+VENT+HBDIFF+ANTHRO` (320.0).

Disclosure (for `02_methods.md`, verbatim):

> Configurations for Aim 2 were selected after the Aim 1 runs had been completed and their descriptive results (mean RMSE per configuration) inspected, but before any Aim 2 model was trained and before any Aim 1 statistical test was run. Three configurations were carried forward, ranked on mean RMSE in the pooled leave-one-subject-out analysis (27 participants): work rate alone (`WR`) as the reference; the configuration without work rate with the lowest RMSE (`HR+VENT+TSI+ANTHRO`); and the configuration combining work rate with at least one physiological domain (HR, VENT or NIRS) with the lowest RMSE (`WR+VENT`). The selection used the same participants as Aim 2, so the absolute errors of the selected configurations may be slightly optimistic. The Aim 2 comparisons (seen vs unseen conditions) are made within each configuration.

---

## 13. Reproducibility, provenance and compute

- **Feature cache:** `outputs/cache/processed/<key>/visits.npz` + `meta.json`. Key = SHA-256 of `PrepSettings`, the source code of `pipeline/prepare/` and `inventory/`, the per-visit HR sources and the overrides. Validity verdicts are joined at load, never baked in. Raw files are assumed immutable (not hashed).
- **Manifest per fold:** git commit + dirty flag, full resolved `Run`, `PrepSettings`, cache key, SHA-256 of `inclusion.csv`, seeds, device, package versions, start/end time, per-epoch wall time, scaler parameters, training-set sizes, checkpoint epochs, packing layout, dropout seed.
- **Environment:** Python 3.11, PyTorch 2.14.1+xpu, NumPy 2.4.6, pandas 3.0.6, pyarrow 25.0.1, pingouin 0.7.0, SciPy 1.17.1, statsmodels 0.15.0 (`requirements-pipeline.txt`). Legacy TensorFlow code untouched (`requirements.txt`).
- **Hardware:** one desktop PC, Intel Arc B580 GPU (XPU), one process. Measured: ≈ 28 ms per step at K = 310 (≈ 1.4 h per epoch for all Aim 1 folds, ≈ 1.7 days at 30 epochs; K = 230 is cheaper). Smoke: ≈ 12 ms/step at K = 2.
- **Tests:** pytest suite (synthetic, device and real-data tests) passes with 0 failures and 0 skips; every proposal acceptance check is mapped to a test (`MAPPING.md` §5): no leakage, fold integrity, one procedure, identical data across configurations, 46 configurations, HRR, causality, seed reproducibility, inclusion snapshot, packed ↔ standalone parity, breath cleaning vs legacy, rm-BA and rm_corr vs legacy.

---

## 14. Departures from the approved Aim 1 proposal

Each was decided by the developer and is recorded in `MAPPING.md`. The paper must report the ones marked ✱.

| Topic | Proposal | Implemented | Why |
|---|---|---|---|
| ✱ Training-settings pilot splits | 3 pooled + 2 × 2 within-cohort validation splits; no test participants | LOSO (54 splits); held-out = Aim 1 test participants | dev, 9 Oct: same scheme as Aim 1 |
| ✱ Pilot decision | rule fixed before running; pilot picks the setting | pilot stopped after 8 pooled splits, no setting chosen; primary = proposal default (5e-5 × 30), sensitivity = 5e-4 × 1, picked after seeing partial pilot results (§9) | dev, 9 Oct: arms within seed noise; pilot cost ≈ Aim 1 itself |
| ✱ NIRS in configurations | TSI and Hb-diff separate, combinable → 62 | one NIRS domain, TSI *or* Hb-diff → 46 | dev, 8 Oct |
| ✱ Shared mask | whole 187-s window valid for every sensor | target needs a valid unit; context needs only present data | lets valid repetitions next to an invalid one keep their targets |
| ✱ Validity | visit-level, from annotations | unit × sensor group, automatic rules + NIRS review | objective, reproducible criteria |
| ✱ HR source | Tymewear | Tymewear, else CPET chest strap (5 visits) | recovers 5 visits' HR |
| ✱ Hb-diff channel | mean of 3 channels | Tx3 (intended 40 mm) | matches Gies's original choice |
| ✱ Fit-factor threshold | 99.9 % | ≥ 99 % in ≥ 95 % of a unit's samples | 99.9 % per sample would drop 9 Lucas visits |
| ✱ SMM | current code | Fitdays skeletal-muscle % × weight (not "Muscle mass") | true SMM |
| ✱ Shuffle | silent (legacy thesis model: off) | on | dev, 8 Oct |
| ✱ Framework | TensorFlow | PyTorch with Keras defaults pinned | TF cannot reliably use the B580 |
| Pilot configurations | `WR`, `HR+VENT`, `WR+HR+VENT+TSI+HBDIFF+ANTHRO` | five (§9) | third is impossible under 46 |
| Pilot size | ≈ 210 trainings | 162 jobs × 10 models | LOSO |
| Seeding | seed Python, NumPy, framework | explicit generators only | — |
| Config / CLI | YAML + command line | frozen dataclasses in `runs.py`, `main.py` | single PC, no cluster |
| Inventory CSV | one table | `inclusion.csv` (verdicts) + `targets.csv` (seconds) | — |

---

## 15. Pending decisions

| # | Item | Options / notes | Blocks |
|---|---|---|---|
| P2 | Statistics (§12) | **deferred** beyond the current effort (dev, 13 Oct 2026), not dropped. Open when resumed: random-effects structure; TOST margin confirm | inferential results (the paper reports descriptives until then) |
| P4 | Tx3 = 40 mm | check OxySoft optode template or ask Artinis; or switch to `mean3` | — |
| P7 | Details not in the repo: pre-test instructions for Gies (Lucas: no heavy exercise/alcohol 24 h, no caffeine that day, 2 h fasting); cart chest-strap model; occlusion protocol for Gies; ergometer and Artinis company cities | ask the cohort owners | Methods text |
| P8 | Run `per_config` sensitivity runs? | optional | — |
| P10 | No TensorFlow ↔ PyTorch numerical check of the ported TCN | add one or state as limitation | — |

---

## 16. Known limitations and caveats

- **Pilot on test participants** (§9, §14): the sensitivity setting (5e-4 × 1) was picked after seeing partial LOSO pilot results, i.e. with test-fold information. The primary setting is the proposal's default and is unaffected.
- **Hb-diff:** raw offset per export; no baseline correction (visual check only); Tx3 distance unverified.
- **Fit factor** has no published threshold; the rule is empirical.
- **Validity vs features use slightly different definitions** in places: inventory Hb-diff gap check uses the 3-channel mean while the feature is Tx3; inventory gap rule counts empty seconds while the presence rule uses sample spacing (≈ 1 s difference); NIRS time base inferred vs header rate. None changes a verdict on current data, as far as checked.
- **Cleaning bridges holes:** after outlier rejection, NaN medians are linearly bridged; only raw-breath spacing can mark a second absent.
- **Shared mask** removes a few targets even for configurations that do not use the missing group (e.g. last seconds of VENT in 10 Lucas visits).
- **Partial visits:** G04 Hard and G07 Medium contribute one repetition each.
- **Same-day visits (Gies):** 10 pairs of visits were performed on the same day, 2.2–7.6 h apart. Prior exercise can alter V̇O₂ kinetics, so these second visits may differ physiologically; it affects data, not the pipeline.
- **Intensity prescription** used 10-s averaged ramp data and a visual GET; relative intensities may not be equivalent across participants.
- **Determinism** holds on the same device and software only.
- **Small cohorts** (27 participants; 12–26 training participants per fold), young healthy adults, laboratory cycling only.
- **BIA-derived SMM** has limited accuracy.
- **Data protection:** Lucas raw files tracked in git include CPET headers with names and dates of birth; the repository remote is public.

---

## 17. Change log

| Date | Change |
|---|---|
| 9 Oct 2026 | Document created from the code at `7bd3739`, `MAPPING.md`, the spec, the review and progress logs. Pilot running. |
| 9 Oct 2026 | Target journals noted (JSHS or Sports Medicine). Participant characteristics computed (§2.1) from the inventory and the per-participant ramp sheets. Protocol levels verified against the xml files: Gies Hard low = GET (P5 closed). Intensity derivation, visit spacing and lab conditions added (§2.2). L14 Interval: wrong CPET profile confirmed by Lucas, age override sufficient (P3 closed). Target-second totals added (§4.4; P11 closed). Same-day visit and intensity limitations added. |
| 11 Oct 2026 | Pilot stopped after 8 pooled splits without a choice (§9, outcome and partial results added). P1 closed: primary 5e-5 × 30 (proposal default), sensitivity 5e-4 × 1 runs added (§8, §10). All six A1 runs complete (§10 status and wall times). §14 pilot-decision row and §16 pilot caveat updated. |
| 13 Oct 2026 | Aim 2 approved (supervisor meeting) and specified, not implemented: Appendix B rewritten from the placeholder (2a without the Ramp, 2b, 2c with a same-test-cohort reference; seen arm size-matched by random subsampling of training targets). Carry-forward rule and disclosure added to §12 (`WR`, `HR+VENT+TSI+ANTHRO`, `WR+VENT`); the "simpler within the margin" idea dropped. P9 closed: the subsampling is specified (Appendix B.4). It replaces the spec's `Subsample(fraction, seed)`, which `MAPPING.md` (`folds.py` row, T08 row) still wrongly lists as implemented; to be corrected with the Aim 2 rows there. §1 and §6 updated. |
| 13 Oct 2026 | Aim 2 implemented (Appendix B status and code references): `pipeline/aim2.py`, `pipeline/aim2_outcome.py`, Aim 2 registry in `pipeline/runs.py`, `tests/test_aim2.py`; pytest passes, smoke run done. The outcome pairing also requires equal `n_s` per visit (B.6). |
| 13 Oct 2026 | Inferential statistics (LMM, Holm, TOST) deferred beyond the current effort (dev); descriptives only for now (§12; P2 marked deferred, not dropped). Residual analyses specified in the new §11.6 (a subsection, so §12–§17 keep their numbers): `A1_pooled` × the three carry-forward configurations × 7 protocols; phase labels from the ideal WR and `inventory/reference/get_power.csv` (provenance added; no MRT shift); 8 disjoint phase windows; joined runs test, per-visit run length, per-phase bias and error share, aligned traces. B.9 points to §11.6; Aim 2 reuse not yet specified. Header stage updated. |

---

## Appendix A: the 46 configurations

In generation order (size, then canonical token order). Input dimension F in brackets.

**1 domain (5):** `WR` [1], `HR` [2], `VENT` [2], `TSI` [1], `HBDIFF` [1]

**2 domains (14):** `WR+HR` [3], `WR+VENT` [3], `WR+TSI` [2], `WR+HBDIFF` [2], `WR+ANTHRO` [3], `HR+VENT` [4], `HR+TSI` [3], `HR+HBDIFF` [3], `HR+ANTHRO` [4], `VENT+TSI` [3], `VENT+HBDIFF` [3], `VENT+ANTHRO` [4], `TSI+ANTHRO` [3], `HBDIFF+ANTHRO` [3]

**3 domains (16):** `WR+HR+VENT` [5], `WR+HR+TSI` [4], `WR+HR+HBDIFF` [4], `WR+HR+ANTHRO` [5], `WR+VENT+TSI` [4], `WR+VENT+HBDIFF` [4], `WR+VENT+ANTHRO` [5], `WR+TSI+ANTHRO` [4], `WR+HBDIFF+ANTHRO` [4], `HR+VENT+TSI` [5], `HR+VENT+HBDIFF` [5], `HR+VENT+ANTHRO` [6], `HR+TSI+ANTHRO` [5], `HR+HBDIFF+ANTHRO` [5], `VENT+TSI+ANTHRO` [5], `VENT+HBDIFF+ANTHRO` [5]

**4 domains (9):** `WR+HR+VENT+TSI` [6], `WR+HR+VENT+HBDIFF` [6], `WR+HR+VENT+ANTHRO` [7], `WR+HR+TSI+ANTHRO` [6], `WR+HR+HBDIFF+ANTHRO` [6], `WR+VENT+TSI+ANTHRO` [6], `WR+VENT+HBDIFF+ANTHRO` [6], `HR+VENT+TSI+ANTHRO` [7], `HR+VENT+HBDIFF+ANTHRO` [7]

**5 domains (2):** `WR+HR+VENT+TSI+ANTHRO` [8], `WR+HR+VENT+HBDIFF+ANTHRO` [8]

---

## Appendix B: Aim 2 design

**Status:** approved 13 Oct 2026 (supervisor meeting) and implemented the same day, additive to the Aim 1 code: specs and registry in `pipeline/runs.py` (`LeftOutArm`, `CohortTransfer`, `A2A_ARMS`, `A2B_ARMS`, `A2C_TRANSFERS`, smoke specs), training in `pipeline/aim2.py`, the paired outcome in `pipeline/aim2_outcome.py`, tests in `tests/test_aim2.py`. Entry points in `main.py`. Every Aim 2 fold folder has the Aim 1 layout, so `evaluate(spec.run)` summarises it unchanged.

### B.1 Questions

Each question compares a model that never saw a condition in training with one that did. Both are tested on the same seconds.

- **2a, unseen intensity (Gies):** one PRBS intensity level (Easy, Medium or Hard) left out of training.
- **2b, unseen protocol (Lucas):** one protocol (Ramp, PRBS, Interval or Step) left out of training. All four are left out in turn and reported together. The Ramp is not singled out as a separate analysis.
- **2c, unseen cohort:** trained on one cohort and tested on the other, in both directions.

### B.2 Common settings

- **Configurations:** the three carried forward from Aim 1 (§12): `WR`, `HR+VENT+TSI+ANTHRO` and `WR+VENT`. They train as one packed job per fold (K = 15).
- **Seeds:** 0–4, none selected.
- **Training:** `A1_TRAINING` (5e-5 × 30) and the §8 procedure, unchanged.
- **Data:** default `PrepSettings` and the shared mask, so every Aim 2 model is scored on exactly the seconds Aim 1 scored. The `A1_*` outputs therefore remain valid references.
- Dropout draws belong to the packed job (§8). A K = 15 job therefore differs from the K = 230 Aim 1 job by seed-level noise only.

### B.3 2a and 2b: unseen and seen arms

For each left-out condition C, there is one fold per test participant who has a C visit. The other participants train it, in one of two arms:

| Arm | Training data (other participants only) |
|---|---|
| **Unseen** | their included visits of every condition except C |
| **Seen** | their included visits of all conditions, subsampled to exactly the unseen arm's number of training targets in that fold (B.4) |

- **2a pool:** the Gies PRBS visits (Easy, Medium, Hard). The **Gies Ramp is never in training** in either arm, for two reasons: the three levels share identical PRBS timing, so 2a then tests the intensity level alone; and only 9 of 14 participants have a Ramp.
- **2b pool:** all Lucas visits (Ramp, PRBS, Interval, Step).
- Participants who lack C still train the folds of the others.
- **Scoring:** only the test participant's C visit is scored. Their other visits are predicted but not used.

| Study | C | Test participants | Pool target-seconds without C / with C | Approx. fraction |
|---|---|---|---|---|
| 2a | Easy | 12 | 47,978 / 71,008 | 0.68 |
| 2a | Medium | 13 | 47,029 / 71,008 | 0.66 |
| 2a | Hard | 13 | 47,009 / 71,008 | 0.66 |
| 2b | Ramp | 13 | 72,388 / 80,449 | 0.90 |
| 2b | PRBS | 13 | 56,354 / 80,449 | 0.70 |
| 2b | Interval | 13 | 56,303 / 80,449 | 0.70 |
| 2b | Step | 13 | 56,302 / 80,449 | 0.70 |

The fractions above are over the whole cohort (`A1_pooled/summary/targets.csv`). The exact counts are computed per fold, without the test participant. G04 Hard and G07 Medium are partial visits (one repetition each) and are scored like any other C visit.

### B.4 Subsampling of the seen arm

- **Target:** n = the unseen arm's number of training targets in the same fold, under the shared mask. The fraction is therefore exact per fold and per C, and no seen arm is shared across conditions.
- **Draw:** exactly n target seconds, uniformly at random without replacement from the seen arm's pool. The draw is not stratified by condition: with n ≈ 44–67 k, each condition's share deviates negligibly from its pool share.
- **One draw per fold**, seeded from (run, fold) through `provenance.derive_seed`. It is shared by all configurations and seeds of that fold, so the seed SD does not include subsampling variation. Each fold gets its own draw.
- **Scalers** are fit after subsampling, on the subsample (§6 rules: feature scalers on the rows the kept windows read, the V̇O₂ scaler on the kept targets).
- **Windows are unchanged:** a kept target reads its full 187-s context, including seconds whose own targets were dropped.
- **Caveat:** windows overlap at 1 Hz, so the seen arm still reads almost every input second of its pool. Size matching therefore equalises the number of training windows (and gradient steps per epoch), not the distinct input content. Contiguous-block subsampling was considered and not adopted (dev, 13 Oct). Revisit it if reviewers ask.

### B.5 2c: unseen cohort

- **Training:** one training job per direction. All included visits of all participants of cohort A (Gies including its Ramp) train the model, which is tested on every participant of cohort B. This needs a multi-participant test fold. Outputs stay one folder per test participant, as in Aim 1.
- **Reference:** the same test cohort's within-cohort LOSO from Aim 1: `A1_lucas` for Gies → Lucas, `A1_gies` for Lucas → Gies. Same configurations, seeds and test seconds; nothing is retrained. The pooled model scored per cohort (`A1_pooled`, cohort groups) is a supplementary reference only.
- **No size matching:**

  | Direction | Cross-cohort training targets | Reference training targets per fold |
  |---|---|---|
  | Gies → Lucas | 76,933 | ≈ 74,300 |
  | Lucas → Gies | 80,449 | ≈ 71,400 |

  The cross-cohort model gets 4 % and 13 % more data. That bias works against finding a generalisation loss, so it is conservative. The sizes are reported.
- **Test visits:** the primary result covers all of the test cohort's visits. The Ramp is the only protocol both cohorts share, so on the other protocols an unseen cohort and an unseen protocol are confounded. Report a per-protocol breakdown, with the Ramp as the cohort-only contrast, and state the confound.

### B.6 Outcome

- **Unit:** participant.
- **Value:** the seed-averaged RMSE (§11.1, steps 1–3) on the C visit (2a/2b), or the mean over the participant's visits (2c).
- **Comparison:** unseen − seen (2c: cross-cohort − within-cohort reference) in mL·min⁻¹, per C (or direction) × configuration. Reported as the mean with a 95 % t-CI across participants. Visits are paired on (configuration, visit, number of target seconds); any unpaired row stops the table (`aim2_outcome.paired_visits`).
- No further inferential modelling (cf. P2).

### B.7 Input scaling

The scalers are unchanged from §6: WR min–max, sex unscaled, other features z-scored, all fitted on the fold's training rows. When the Ramp is left out in 2b, the test WR exceeds the training maximum and scales to > 1. That is the extrapolation case. 2a stays within the training range, because Easy and Hard both lie inside Medium's 25 W ↔ Δ30 % span.

### B.8 Runs and compute

| Runs | Fold jobs |
|---|---|
| `A2a_{easy,medium,hard}_{unseen,seen}` | (12 + 13 + 13) × 2 = 76 |
| `A2b_{ramp,prbs,interval,step}_{unseen,seen}` | 13 × 4 × 2 = 104 |
| `A2c_gies_to_lucas`, `A2c_lucas_to_gies` | 2 |

### B.9 Supporting analyses

The temporal structure of the residuals (runs tests, run lengths) and the error by exercise phase are specified in §11.6, on the Aim 1 run `A1_pooled`. Their reuse for the Aim 2 run IDs (the same analyses on other runs, a seen-vs-unseen comparison of run lengths) is not yet specified.

### B.10 Changes from the handoff's Aim 2 sketch (handoff §6)

- **2a:** the Ramp is excluded from training. The sketch did not exclude it.
- **2c:** the sketch's "that cohort's within-cohort LOSO" is made explicit: the reference is the **test** cohort's, not the training cohort's. Otherwise the comparison mixes "trained elsewhere" with "this cohort is easier or harder to predict".
- **Subsampling:** an exact count per fold as in the sketch, drawn once per fold. It is not stratified by protocol (`MAPPING.md` had described stratification).
- **Configurations:** the sketch did not limit them. Aim 2 uses the three carried forward (§12).
