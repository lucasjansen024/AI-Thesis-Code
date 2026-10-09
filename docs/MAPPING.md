# MAPPING: Aim 1 handoff → harmonised pipeline

This file maps every requirement in [`AIM1_IMPLEMENTATION_HANDOFF.md`](AIM1_IMPLEMENTATION_HANDOFF.md) §2–§7 to the place in the new `pipeline/` package where it lands, and gives each one a status. It is handoff §4 step 1, and it is the input for the implementation spec. Terms (*unit*, *sensor group*, *shared mask*, *NIRS variable*, *trajectory* …) are defined in [`CONTEXT.md`](../CONTEXT.md).

**Status (8 Oct 2026):** every row is decided. No pipeline code exists yet. `inventory/` and `inventory/inclusion/inclusion.csv` are built and committed.

**Status values**

| Status | Meaning |
|---|---|
| **as-is** | Legacy already met the requirement; the new code carries the behaviour over. |
| **changed** | The handoff required a change from legacy; the new code makes it as specified. |
| **departure** | The new code differs from the handoff. Each one was decided by the dev; the one tagged **needs-supervisor** still needs sign-off. |
| **addition** | Beyond the handoff; changes no choice the handoff makes. |

**Decided in** cites the dev's local planning tickets (not in the repo):

| # | Ticket |
|---|---|
| T01 | Bring Gies's data into the workspace |
| T02 | PortaMon signal semantics |
| T03 | Training platform and speed levers |
| T04 | Data inventory of both cohorts |
| T05 | Data validity and inclusion rules |
| T06 | Harmonised preparation and feature definitions |
| T07 | Training-speed prototype and compute budget |
| T08 | Pipeline architecture, experiment registry and feature cache |
| T09 | Training procedure and pilot design |
| T10 | Run outputs and evaluation metrics |
| T11 | Final NIRS review calls |
| T12 | Build the inclusion CSV |
| T13 | Acceptance tests and smoke run |

---

## 1. Module map

`pipeline/` is a new top-level package next to `inventory/`. Legacy `src2_training_ready/` stays untouched (§9). File names inside `prepare/`, and the `metrics.py`/`evaluate.py` split, are proposals for the spec; every other name was decided in T08–T10.

| Module | Role |
|---|---|
| `prepare/` | Raw → one processed 1 Hz table per visit. Loads only through `inventory.catalog`/`inventory.sensors`. Proposed files: `grid.py` (1 Hz grid, `phase`, `unit`), `wr.py`, `breath.py` (§2.7), `hr.py`, `nirs.py`, `anthro.py`, `visit.py` (assembles the table and the presence flags). |
| `cache.py` | `outputs/cache/processed/<key>/visits.npz` + `meta.json`. Key = hash of `PrepSettings` + source hash of `pipeline/prepare/` and `inventory/`. `rebuild=True` forces a rebuild. |
| `inclusion.py` | Reads `inventory/inclusion/inclusion.csv` with a schema check; builds the target mask (shared or `per_config`). |
| `domains.py` | `Domain(StrEnum)`: WR, HR, VENT, NIRS, ANTHRO; NIRS variables TSI/HBDIFF; feature columns per domain; configuration generator and IDs. |
| `folds.py` | `Fold(test_participant, train_visits)`, LOSO generation, `Subsample(fraction, seed)`, fold scalers, target indices. In memory only. |
| `TCN.py` | Packed §2.3 TCN (grouped convs, per-branch LayerNorm, channel masks, last-only forward) and extraction into standalone per-configuration TCNs. |
| `train.py` | One job per (fold, identical-mask group): seeds, on-device window gather, checkpoint, `.tmp` → final rename, resume. |
| `runs.py` | Frozen `Run` dataclasses (`A1_pooled`, `A1_gies`, `A1_lucas`, `per_config` variants, `SMOKE`), pilot splits, the chosen training setting as a literal. |
| `pilot.py` | Pilot trajectories with validation logging; `decide()`. |
| `metrics.py` | Per-visit metrics, `rm_corr`, Pearson r, repeated-measures Bland–Altman, VO₂peak. |
| `evaluate.py` | `evaluate(run)`: seed aggregation, `summary/` tables. |
| `main.py` (repo root) | Calls `run(...)`, `evaluate(...)`, pilot; runs picked by commenting/uncommenting. |

**Data flow:** raw → `prepare` → cache npz → join `inclusion.csv` → mask → fold (scalers, target indices) → packed training → standalone models → `outputs/runs/<run_id>/<participant_id>/` → `evaluate` → `summary/`.

---

## 2. Required pipeline changes (handoff §2)

### §2.1 Scaling

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| Fit every scaler (inputs and V̇O₂) on the fold's training participants only | `folds.py` fold scalers | T08, T13 | as-is | Legacy `get_fold_data` was already fold-wise. Runtime check: scaler-fit rows contain no test-participant rows (explicit exception, not `assert`). |
| Inverse-transform predictions to mL·min⁻¹ before any metric | `train.py` (prediction), `metrics.py` | T10 | as-is | Metrics in float64 on mL·min⁻¹. |
| WR min–max; other continuous features z-score; sex 1/0 unscaled | `folds.py` | T06 | as-is | Includes TSI, Hb-diff, SMM, HRR as z-scored. |
| No global scaling anywhere | `folds.py`; test | T13 | as-is | Legacy global path (`__create_scalars`) is unreachable; the new code has none. Test: extreme test-participant values leave every fold scaler unchanged. |

### §2.2 Seeds

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| 5 seeds (0–4) per configuration × fold | `runs.py`, `train.py` | T07, T09 | changed | All 5 seeds packed in one job per fold, one batch stream per seed. |
| Seed Python, NumPy and the framework; op determinism | `train.py` | T09 | departure (framework) | PyTorch instead of TensorFlow. Seed `s` fixes init (per-branch generator from `(s, config_id)`) and the per-epoch permutation. Dropout comes from one RNG per job. `torch.use_deterministic_algorithms(True)` (~1 % cost); `CUBLAS_WORKSPACE_CONFIG` on CUDA. |
| Save every run; never select among seeds | `train.py` → `models.pt`, `predictions.parquet` | T10 | changed | All 5 seeds saved; no seed selection anywhere. |
| Mean ± SD across seeds; store the seed-averaged prediction per second | `evaluate.py` → `summary.csv`, `predictions_seedavg.parquet` | T10 | changed | *Seed-averaged metric* = mean of per-seed metrics, not the ensemble's metric. |

### §2.3 One training procedure

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| TCN: 5 causal dilated Conv1D, k 7, 16 filters, dilations 1–16, residual, LayerNorm, ReLU, dropout 0.2; RF 187 s | `TCN.py` | T07, T08 | departure (framework) | PyTorch port with Keras defaults pinned (Glorot kernels, zero biases, LayerNorm eps 1e-3). Training uses the last-only forward: same function and gradients (parity-tested). Unused feature channels are masked; Glorot uses the true fan-in. Evaluation uses extracted standalone models. No TF ↔ PyTorch numerical parity test exists. |
| Adam; lr 5e-5 default, final by pilot | `train.py`; literal in `runs.py` | T07, T09, T10 | as-is | Adam eps 1e-7 (Keras). The dev copies the pilot's lr into `runs.py`. |
| Batch 32 | `train.py` | T08 | as-is | Last partial batch kept (Keras behaviour). |
| Epochs 30 default, final by pilot | literal in `runs.py` | T09, T10 | as-is | |
| Loss MSE | `train.py` | T07 | as-is | Packed loss = sum of per-model MSEs, so each model gets its own gradient. |
| Checkpoint: lowest **training** loss | `train.py` | T09 | as-is | Keras-style epoch average (batch-size weighted, dropout active), strict `<`, per branch, held in memory. |
| No validation set | `train.py` | T09 | as-is | |
| Shuffle | `train.py` | T09 | departure | **On** (seeded, new permutation each epoch). Legacy thesis model had `shuffle=False`; the handoff is silent, so the default would have been "off". Dev, 8 Oct. |
| Same procedure for pooled, within-cohort and later Aim 2 runs | `runs.py` (one training-settings literal) | T08, T13 | as-is | Test: resolved training settings identical across A1 manifests. `evaluate()` checks fold manifests within a run agree. |
| Never use test data to pick an epoch, checkpoint or setting | `train.py`, `pilot.py` | T09; dev, 9 Oct | **departure** | Checkpoint on training loss (met). The setting is not: the LOSO pilot chooses it on the A1 test participants (§3.0 below). |

### §2.4 Windows and evaluation segment

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| Window for target `t` is `t−187 … t−1` | `train.py` window gather; `inclusion.py` mask | T08, T13 | as-is | Runtime check: last gathered index = `t−1` (vectorised per fold). Test: perturbing inputs ≥ `t` leaves the output unchanged. |
| Targets only from exercise; warm-up end is context; rest excluded | `prepare/grid.py` (`phase`), `inclusion.py` | T05, T06 | as-is | Flag `warmup_as_context=true`. Phase boundaries from the nominal timeline. Rest never enters a window (earliest context second 143 s). Ramp end = last breath with cadence ≥ 60 rpm. |

### §2.5 HRR

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| `HRR = (HR − HRrest) / (HRpeak − HRrest)` | `prepare/hr.py` | T06 | changed | Legacy used min/max across visits and baked HRR into prepared files. |
| HRrest = mean HR in the same visit's pre-exercise rest | `prepare/hr.py` | T06 | changed | 0–120 s of the same HR source; flag `hrrest_window_s=120`. A visit without rest HR coverage (< 80 %) fails the HR group in `inclusion.csv`. |
| HRpeak = `208 − 0.7 × age` for all 27 | `prepare/hr.py` | T06 | changed | Age = that visit's CPET header age, full precision. |
| Age available for all 27 participants | `inventory` (checked) | T04 | as-is | Present in every visit. |
| HR source | `prepare/hr.py`; `hr_source` from `inclusion.csv` | T05 | departure | Tymewear if it passes the HR check, else the CPET chest strap (handoff §1/§3.1 list Tymewear only). Affects G06 Ramp/Easy, G09 Ramp, G11 Easy, G13 Medium. |

### §2.6 NIRS

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| Features TSI and Hb-diff, not normalised; no occlusion or within-visit min–max | `prepare/nirs.py` | T06 | changed | Hb-diff = `O2Hb3 − HHb3` (Tx3); see §7 for the channel. |
| 10 Hz → 1 Hz by averaging each second | `prepare/nirs.py` | T06 | changed | Second `t` = mean of samples in `[t, t+1)`. |
| Only scaling = fold z-score | `folds.py` | T06 | changed | |
| Report the PortaMon zero point before running | this file (below) | T02, T06 | changed | **Report:** no source says when OxySoft sets the zero; raw Hb-diff has an arbitrary per-visit (even per-export) offset. The dev's plots show rest-mean subtraction makes barely a difference, so raw stays the default (§7). |
| NIRS loader fixes (not in the handoff) | `prepare/nirs.py` | T02, T06 | addition | Drop all-zero rows and the nonphysical first row; read "Export sample rate" from the header; sync on the **latest** `XminYY` marker. |

### §2.7 Breath-by-breath cleaning

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| Reject > 3 SD vs previous 5 breaths → 5-breath median → interpolate to 1 Hz; V̇O₂, V̇E, BR; same for both cohorts | `prepare/breath.py` | T06 | as-is | Exact port of legacy `reject_outliers_pre_local_mean` + `median_filter` (centred for V̇O₂, trailing for V̇E/BR), vectorised. Input: CPET V̇O₂ and the raw Tymewear `VE/BR breath by breath` columns. Test against legacy output (real data). |

### §2.8 Data inclusion and the shared mask

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| Repeated visits: use the complete one | `inventory/catalog`, `inclusion.csv` | T04, T05, T12 | as-is | G08 Medium2 and G09 Hard2 used (`source_visit`); superseded visits and G13 FailedRamp left out of the CSV. Medium2/Hard2 reuse the sibling's protocol xml. |
| Shared mask: only visits where WR, Tymewear and NIRS are valid | `inclusion.py` | T05, T08 | departure | Validity is per **unit** (PRBS repetition, else the visit's exercise phase) and per **sensor group** (VO₂, WR, HR, VENT, NIRS, static). The shared mask ANDs all groups. |
| Window-level mask: target counts only if its full 187 s window is valid for every sensor | `inclusion.py` | T05, T08 | departure | Replaced: a target needs a valid unit; its 187 s context is admitted whatever its own validity, provided every required input group is **present** (no missing source, no gap > 20 s). G07 Medium rep 2 keeps all its targets, with rep 1 as context. On current data the presence rule never drops a target. |
| Validity rules (handoff silent on how) | `inventory/build_inclusion.py` → `inclusion.csv` | T05, T12 | addition | Layer 1 (automatic): missing source; gap > 20 s on raw timestamps (`max_interp_gap_s=20`); HR span ≥ 20 bpm and r vs smoothed V̇O₂ ≥ 0.5 from rest to exercise end, with rest coverage; NIRS fit ≥ 99 % in ≥ 95 % of seconds. Layer 2: the dev's NIRS review list (`inventory/inclusion/nirs_review.csv`). Gies annotations retired for validity. |
| Expected Gies exclusions; verify against raw | `inclusion.csv`; snapshot test | T04, T05, T11, T12, T13 | departure | Gies **47 of 55** (handoff: ~44 of 56). Differences in §8. |
| All 52 Lucas visits valid; verify | `inclusion.csv` | T04, T12 | as-is | Verified: 52 of 52. |
| Inventory CSV: participant, cohort, protocol, sensor, valid seconds, included, reason | `inventory/inclusion/inclusion.csv` + `summary/targets.csv` | T05, T10, T12 | departure (form) | Split in two: `inclusion.csv` holds verdicts per unit × sensor group (`valid`, `reason`, `decided_by`, `hr_source`); `targets.csv` holds exercise and target seconds per visit × unit after the mask. Together they feed the data-flow figure. Exploratory inventory in `outputs/inventory/`. |
| Sensitivity flag `mask: per_config` | `Run.mask`; `inclusion.py`; separate run_ids | T05, T08 | as-is | Uses the same unit table, ANDing only VO₂ + the configuration's groups. Configurations with identical target sets share a job (≤ 8 groups per fold). |

### §2.9 IDs and grouping

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| Unique `participant_id` and `cohort` label | `inventory/catalog` → processed table | T04 | changed | `G01, G03–G15`, `L01–L10, L12–L14` (original numbers kept). Plus `visit_id` (`G08_Medium`) and `source_visit` (`Medium2`). |
| Splits by participant | `folds.py` | T08, T13 | as-is | Runtime check: test ∉ train, unique test IDs. |
| "sex", not "gender" | `prepare/anthro.py`, all outputs | T06 | changed | Sex from the CPET header, male = 1. |

---

## 3. Aim 1 experiments (handoff §3)

### §3.0 Training-settings pilot

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| Arms R, A, B, C, D (lr × epochs) | `pilot.py` | T09 | as-is | Trained as 3 **trajectories** (5e-5 × 30, 2e-4 × 10, 5e-4 × 3); A is read off R and C off D. Same procedure: constant lr, no schedule. Each arm is scored at `e* = argmin_{e ≤ E}` training loss. |
| Log training and validation loss per epoch | `pilot.py` → `outputs/pilot/<split>/<trajectory>/epochs.csv`, `val_visits.csv` | T09, T10 | as-is | |
| 3 pooled splits (3 participants, mixed cohorts); 2 within-cohort splits per cohort (2 participants) | `runs.loso_pilot_splits()`; `pilot.check_loso_coverage` | dev, 9 Oct | **departure** | **LOSO instead of validation splits:** every family holds out each of its participants once (27 pooled, 14 Gies, 13 Lucas splits), exactly like the A1 folds, so pooled and within-cohort pilot results use the same scheme as Aim 1. Consequence: each held-out participant is an A1 test participant, so the setting is chosen on test-fold results (§2.3 and §3.0 forbid this; the methods section must say so). |
| Configurations `WR`, `HR+VENT`, `WR+HR+VENT+TSI+HBDIFF+ANTHRO` | `runs.py` | T09; dev, 9 Oct | departure | Replaced by 5: `WR`, `WR+HR+VENT+TSI+ANTHRO`, `WR+HR+VENT+HBDIFF+ANTHRO`, `HR+VENT+TSI+ANTHRO`, `HR+VENT+HBDIFF+ANTHRO`. The original third configuration cannot exist under 46 configurations. The dev ruled that no supervisor sign-off is needed. |
| 2 seeds | `runs.py` | T09 | as-is | Seeds 0, 1. Shared mask. |
| Size ≈ 210 trainings | `pilot.py` | T07, T09; dev, 9 Oct | changed | 162 jobs (54 LOSO splits × 3 trajectories) × K = 10 (5 configurations × 2 seeds) ≈ 27 h on the B580 (pooled fold ≈ 40 min, within-cohort fold ≈ 20 min, per 3 trajectories). |
| Decision rule: cheapest arm within 10 mL·min⁻¹ of the best, everywhere; else R | `pilot.py: decide()` → `decision.json`, `decision.md` | T09, T10; dev, 9 Oct | **departure** | `decide()` still computes the table and the rule's result: 15 cells = {pooled, Gies, Lucas} × 5 configurations; held-out RMSE per participant-visit → participant → splits × seeds; an A/B tie goes to the lower mean RMSE. **The rule is not binding:** the dev picks the setting after seeing the table (the handoff required a rule written down before running). |
| Never use test results | — | dev, 9 Oct | **departure** | Not met: under LOSO, the held-out participants are the A1 test participants (see the splits row). |

### §3.1 Input domains

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| Domain feature sets (WR; HR+HRR; V̇E+BR; TSI; Hb-diff; sex+SMM) | `domains.py` | T06, T08 | as-is | Feature definitions in §2.5, §2.6, §7. VENT is Tymewear only (CPET V̇E/Rf share the flow signal of V̇O₂). |
| TSI and Hb-diff as two separate domains, combinable → 62 configurations | `domains.py`; `Run.nirs_exclusive` | T08 | departure | NIRS is one domain with two alternative **NIRS variables**: **46 configurations** by default (`nirs_exclusive=True`); `False` gives 62. |
| ANTHRO never alone | `domains.py` | T08 | as-is | |
| Generated programmatically; stable readable IDs | `domains.py` | T08 | as-is | Ordered by size, then canonical order: `WR+HR+TSI+ANTHRO`. |

### §3.2 Runs

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| `A1_pooled` (27), `A1_gies` (14), `A1_lucas` (13), LOSO | `runs.py`, `folds.py` | T08 | as-is | Test: 27/14/13 folds, unique test IDs. |
| Size 16,740 trainings | `train.py` | T07, T08 | changed | 46 × 5 × 54 folds = **12,420 trainings in 54 jobs**. Measured budget (B580, K = 310): ≈ 1.4 h per epoch, ≈ 1.7 days at 30 epochs, an upper bound for K = 230. No scope cut needed. |
| Within-cohort runs: all configurations (minimum: those carried forward) | `runs.py` | T07 | as-is | All 46. |
| Pooled first | `main.py` order | T08 | as-is | |
| Command-line subsets of configurations, folds, seeds; job arrays | `main.py` registry; `run(run, folds=...)` | T08 | departure (form) | No YAML, no CLI: frozen `Run` dataclasses, picked by commenting/uncommenting. Subsets are per fold, since one job packs all configurations × seeds of a fold. No cluster, so no job arrays. |
| Resumable: skip runs whose outputs exist | `train.py` | T08 | as-is | Fold writes to `<participant_id>.tmp/`, renamed after `manifest.json`. A finished fold is skipped; a leftover `.tmp` is wiped. |

### §3.3 Outputs per training run

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| Predictions parquet, exact §3.3 columns | `outputs/runs/<run_id>/<participant_id>/predictions.parquet` | T10 | changed | Test participant, target seconds, mL·min⁻¹. Test: columns are exactly the §3.3 list. Adds dependency `pyarrow`. |
| Per-visit RMSE, MAE, bias, n seconds per participant × visit × seed | `visit_metrics.csv` | T10 | changed | Bias = mean(pred − true). |
| Manifest: git commit, resolved config, seed, versions, start/end | `manifest.json` | T10 | changed | Adds: dirty flag, `PrepSettings`, cache key, inclusion CSV hash, device, per-epoch wall time, packing layout, scaler params, training-set sizes. |
| (addition) per-epoch losses, standalone weights | `epochs.csv`, `models.pt` | T09, T10 | addition | `models.pt` ≈ 9 MB per fold; re-evaluation needs no retraining. |

### §3.4 Evaluation and reporting

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| RMSE per participant-visit; per configuration mean ± SD across participants (seed-averaged) and SD across seeds | `evaluate.py` → `summary.csv` | T10 | changed | Participant value = mean of its visits. SD across seeds = SD of the 5 per-seed configuration means. Same for MAE and bias. |
| Repeated-measures correlation (`pingouin.rm_corr`) | `metrics.py` | T10 | as-is | Mean ± SD over seeds; r only (no p-values). Test: matches legacy `Analysis.rm_corr`. |
| Bland–Altman bias and 95 % LoA, repeated measures (B&A 2007) | `metrics.py` | T10 | as-is | Port of legacy `Analysis.bland_altman_global` ("true value varies"); formula correct. Fixed in the port: no input mutation, no `print`, variance component clamped at 0, grouping by exact ID. |
| Breakdowns of the pooled run by cohort and protocol | `evaluate.py` | T10 | as-is | Every run gets groups all / per cohort / per protocol. |
| (addition) Pearson r, VO₂peak on ramps | `metrics.py` → `summary.csv`, `vo2peak.csv` | T10 | addition | VO₂peak = highest 30-s trailing mean (`vo2peak_window_s=30`); legacy used a 1-s max. MNG and the runs test dropped (runs test is Aim 2). |
| Planned statistics: LMM, Holm contrasts vs `WR`, TOST | input table `summary/visit_metrics_seedavg.csv` | T10 | as-is | Statistics scripts are outside this work; only their input table is produced. |
| No winner selection in the pipeline | `evaluate.py` | T10 | as-is | |
| Training-set sizes reported (§2.3) | `summary/folds.csv` | T10 | changed | `n_train_participants`, `n_train_targets`, `n_test_targets`, wall time. |

---

## 4. Implementation order (handoff §4)

| Step | Where it stands |
|---|---|
| 1. `MAPPING.md` | This file. |
| 2. Load Gies like Lucas; IDs; inventory script | Done in `inventory/` (outputs in `outputs/inventory/`, verdicts in `inventory/inclusion/`). |
| 3. HRR, NIRS, shared mask | `prepare/`, `inclusion.py` (§2.5, §2.6, §2.8 above). |
| 4. YAML config | Replaced by the `runs.py` registry and `PrepSettings` (§3.2 above). |
| 5. Training loop | `train.py`, `TCN.py`, `folds.py`. |
| 6. Evaluation script | `metrics.py`, `evaluate.py`. |
| 7. Tests, then smoke run (2 configurations × 2 folds × 1 seed) | §5 below. `SMOKE`: `WR` and `WR+HR+VENT+TSI+ANTHRO`, folds G07 and L04, seed 0, 3 epochs, plus a pilot smoke. |
| 8–9. Pilot, full runs | After implementation; not part of this plan. |

---

## 5. Acceptance checks (handoff §5)

**Done** = the full pytest suite on the B580 PC (with `data/raw/`) gives 0 failures and 0 skips, the smoke-run checklist passes, and the pilot smoke writes its logs. Synthetic fixtures live in `tests/conftest.py`; tests marked `@pytest.mark.real` use the real cache and skip only on a machine without `data/raw/`. Device-dependent tests run on XPU. Runtime checks raise explicit exceptions, never `assert`.

| §5 check | Test | Runtime check too | Data | Decided in |
|---|---|---|---|---|
| No leakage | Extreme test-participant values leave every fold scaler (features and V̇O₂) unchanged | Scaler-fit rows exclude the test participant | synthetic | T13 |
| Fold integrity | 27/14/13 folds, unique test IDs, test ∉ train | Test ∉ train, IDs unique | real | T13 |
| One procedure | Resolved training settings identical across A1 manifests (pilot excluded) | `evaluate()`: fold manifests within a run agree | outputs | T10 |
| Identical data across configurations | Within a fold, the evaluated `(visit_id, t_s)` set is identical for every configuration | Shared mask → exactly 1 mask group | real | T10 |
| Configuration list | **46** unique (62 with `nirs_exclusive=False`), never TSI + HBDIFF, domain features always together | — | none | T08 |
| HRR | Formula exact on synthetic data. Real: per visit, p1–p99 of exercise HRR within [−0.1, 1.1]; failing visits listed | — | both | T13 |
| Causality | Window for `t` is exactly `t−187…t−1`; output at `t` unchanged when inputs ≥ `t` are perturbed | Last gathered index = `t−1` | synthetic | T13 |
| Seeds | Same tiny job on the same device is bit-identical; different seeds give different weights | — | synthetic, XPU | T09 |
| Inventory | Snapshot: `inclusion.csv` yields exactly the frozen list of included units (Gies 47 of 55, Lucas 52 of 52). Differences from §2.8 are explained in §8 below | — | committed CSV | T13 |

**Other tests:** inclusion CSV schema; missing-source verdicts agree with the processed presence flags (real); G07 Medium rep 2 first target `t = 1290` with context 1103–1289 (real); packed ↔ standalone and last-only parity (XPU); §2.7 cleaning matches legacy (real); predictions columns = §3.3; `evaluate` raises on a missing fold; rm-BA port (MS_b > MS_w matches legacy; MS_b < MS_w gives σ² = MS_w); `rm_corr` matches legacy; seed-averaged ≠ ensemble RMSE; `Subsample` keeps the fraction per protocol; cache key changes when settings or source change; resume skips a finished fold and wipes a leftover `.tmp`; `decide()` on hand-built logs (qualify rule, A/B tie, fallback to R).

**Smoke run checklist** (after `run(SMOKE)` + `evaluate(SMOKE)`):
1. Both fold folders and `summary/` exist; no `.tmp` left.
2. `epochs.csv` has 3 epochs per model; `is_best` consistent with the minimum loss.
3. RMSE recomputed from `predictions.parquet` ≈ `visit_metrics.csv`.
4. Rerunning `run(SMOKE)` skips both folds within seconds.
5. Per-epoch wall time within ~2× of the measured budget, scaled to K = 2.

**Pilot smoke:** split `pooled_G01` (G01 held out) × trajectory D (5e-4, 3 epochs) × the same 2 configurations × seed 0. Checks that `val_visits.csv` and `val_mse_scaled` are written; `decide()` is not run.

---

## 6. Aim 2 extensibility (handoff §6)

| Requirement | Lands in | Decided in | Status | Note |
|---|---|---|---|---|
| Training set as an explicit list of (participant, visit) pairs + optional seeded subsample fraction | `folds.Fold(test_participant, train_visits: frozenset[visit_id])`, `folds.Subsample(fraction, seed)` | T08 | as-is | `Subsample` keeps the fraction of target seconds **within each protocol**. Implemented and tested, unused in Aim 1. |
| Protocol label on every saved prediction | `predictions.parquet` | T10 | as-is | |
| Generic `run_id` | `runs.Run` | T08 | as-is | |
| GRU kept, not run | legacy `src2_training_ready/models/` | — | as-is | `TCN.py` is named per model so a GRU module can sit beside it later. |

---

## 7. Open decisions → flags (handoff §7)

| Decision | Flag (where) | Default | §7 default | Status | Decided in |
|---|---|---|---|---|---|
| TSI and Hb-diff in one configuration | `nirs_exclusive` (`Run`) | `True` → 46 | allowed → 62 | departure | T08 |
| Hb-diff channel | `hbdiff_channel: tx3 \| mean3` (`PrepSettings`) | `tx3` | mean of 3 | departure | T06 |
| Hb-diff offset / baseline | `hbdiff_baseline: none \| rest_mean` (`PrepSettings`) | `none` | raw as exported | as-is | T02, T06 |
| HRrest | `hrrest_window_s` (`PrepSettings`) | 120 (0–120 s, same visit, same source) | same-visit rest mean | as-is | T06 |
| SMM | `smm: per_visit \| participant_mean` (`PrepSettings`) | `per_visit` | current code | departure (definition) | T06 |
| NIRS fit-factor threshold | `NIRS_FIT_SHARE_COLUMN = "ff_ge99"` (the 99 % level), `NIRS_MIN_FIT_SHARE = 0.95` (`inventory/build_inclusion.py`) | 99 % in ≥ 95 % of a unit's seconds | 99.9 % | departure | T02, T05, T12 |
| Training setting (lr × epochs) | literal in `runs.py`, chosen by the dev from `outputs/pilot/decision.md` | 5e-5 × 30 until the pilot | pilot rule | departure (dev picks; rule shown, not binding) | T09, T10; dev, 9 Oct |
| Pilot tolerance | constant in `pilot.py` (informational) | 10 mL·min⁻¹ | 10 mL·min⁻¹ | as-is | T09 |
| Equivalence margin | statistics scripts (outside this work) | ±25 mL·min⁻¹ | ±25 mL·min⁻¹ | as-is | — |
| Within-cohort runs | `runs.py` | all 46 configurations | all configurations | as-is | T07 |

**Notes on the departures**
- **SMM:** per-visit as before, but the quantity changes: SMM kg = Fitdays `Skeletal muscle` % × `Weight`, matched to the visit by date (fails loudly on no match). Legacy and Lucas's thesis used Fitdays "Muscle mass", a different, larger quantity.
- **Hb-diff Tx3:** matches Gies's original 40 mm choice. "Tx3 = 40 mm" is unverified on both devices.
- **Fit factor:** the fit factor qualifies TSI only and has no published threshold. Lucas's 99.9 % was a placement check; per sample it would drop 9 Lucas visits entirely. It gates the NIRS group (TSI and Hb-diff together).

**Flags added for gaps** (keep current behaviour or the dev's call): `warmup_as_context=true`, `max_interp_gap_s=20`, `hr_min_span_bpm=20`, `hr_min_r_vo2=0.5`, `hr_min_rest_coverage=0.8` (thresholds in `inventory/build_inclusion.py`), `mask: shared | per_config` (`Run`), `vo2peak_window_s=30`, `rebuild` (cache).

---

## 8. Gies inclusion: handoff §2.8 vs actual

Handoff labels P*n* are participant `G<nn>`. Source: `inventory/inclusion/inclusion.csv` (verified 8 Oct 2026).

| Visit | Handoff §2.8 | Actual (shared mask) | Why |
|---|---|---|---|
| G06 Ramp | excluded (Tymewear) | excluded | No Tymewear file. |
| G07 Ramp | excluded (Tymewear, NIRS) | excluded | No Tymewear file. NIRS judged valid (dev + Lucas, 8 Oct); matters only under `per_config`. |
| G07 Hard | excluded (Tymewear; NIRS 2nd half) | excluded | Tymewear HR fails the HR check (r 0.00), no CPET HR. Tymewear V̇E/BR are valid. |
| G09 Ramp | excluded (Tymewear) | excluded | No Tymewear file. |
| G11 Ramp | excluded (Tymewear, NIRS) | excluded | No Tymewear file. NIRS included by review (`per_config` only). |
| G13 Ramp | excluded (Tymewear) | excluded | Tymewear file present but its HR is empty; CPET HR stuck. Excluded on HR, not VENT. |
| G13 Easy | excluded (Tymewear) | excluded | Tymewear HR missing; CPET HR stuck. Excluded on HR, not VENT. |
| G08 Easy | excluded (NIRS) | excluded | NIRS review. |
| G14 Medium | excluded (Tymewear, NIRS) | — | Never recorded: no folder, no lab-log entry. Gies has 55 candidate visits, not 56. |
| **G07 Medium** | excluded (NIRS) | **rep 2 included**, rep 1 excluded | NIRS review: exercise NIRS is clean, but a warm-up TSI artifact (~150–280 s) would enter rep 1's first windows, so the dev excluded rep 1. Rep 2 keeps all targets, with rep 1 as context. |
| **G11 Easy** | excluded (NIRS) | **included** | NIRS review. |
| **G11 Hard** | NIRS 2nd half missing | **both reps included** | NIRS review: no problem seen in rep 2. |
| **G15 Ramp** | excluded (NIRS) | **included**, whole ramp | NIRS review. Fit factor drops near the end, but the unit passes the ≥ 99 % / 95 % rule; no partial cut. |
| **G04 Hard** | included | **rep 1 only** | CPET recording stops at ~1355 s; rep 2 fails the gap rule on V̇O₂ (gap 894 s). Not in the handoff or the annotations. |
| G08 Medium, G09 Hard | use the complete visit | Medium2, Hard2 used | As specified. |
| G13 FailedRamp | — | not used | Aborted visit, repeated as G13 Ramp. |

**Result:** Gies 47 of 55 visits (G04 Hard rep 1 only, G07 Medium rep 2 only); Lucas 52 of 52. Under `per_config`, `WR` alone uses all 55 Gies visits.

---

## 9. Legacy code

**Untouched.** All of `src2_training_ready/` (preparer, `DataManager`, `Processor`, `Trainer`, `ResultsManager`, `analysis.py`, models including the GRU, `experiments/`), `requirements.txt` (TF 2.15), and `data/prepared/` and `data/prepared_gies2024/` (comparison reference only). The new pipeline replaces `DataManager`, `Processor`, `Trainer` and `ResultsManager`; it does not modify them.

**Reached at run time,** only through `inventory/`:
- `src2_training_ready.loaders.VO2_loader.VO2Loader`, `VE_loader.VELoader`, `NIRS_loader.NIRSLoader` (via `inventory/sensors.py`);
- `src2_training_ready.preparation.config.config_lucas` (via `inventory/annotations.py`);
- `src2_training_ready.preparation.data_preparer.VISIT_ORDER` (via `inventory/metrics.py`).

**Reached by tests only,** as comparison references: `src2_training_ready.analysis.Analysis` (`bland_altman_global`, `rm_corr`) and `src2_training_ready.processing.helper_functions` (`reject_outliers_pre_local_mean`, `median_filter`). `analysis.py` imports `statsmodels` and `scipy`, so the pipeline venv needs both for those tests.

**Legacy faults the new code does not carry over:** first-match file choice (Lucas Ramp 10-s exports, `~$` lock files), ramp WR starting 30 s early (210 s vs 240 s warm-up), silent interpolation across gaps of any length, HRR from min/max across visits, NIRS min–max normalisation, first-marker NIRS sync, no shuffle.

---

## 10. Environment

- One venv: the dev's project venv with `torch 2.14.1+xpu`. `pipeline/` and `inventory/` both run in it.
- New `requirements-pipeline.txt`, pinned from that venv's `pip freeze`: torch (XPU index URL; note on the CUDA wheel for the A500), numpy, pandas, `pyarrow`, `pingouin`, `pytest`, plus `statsmodels` and `scipy` for the legacy-comparison tests. The legacy `requirements.txt` is unchanged.
- Training device: the B580 (XPU), one process. `torch.compile` is an optional flag outside the budget.
