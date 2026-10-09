# Aim 1 implementation handoff: harmonised pipeline + sensor-modality comparison

**Status (7 Oct 2026):** Aim 1 is approved by the supervisor. Aim 2 (generalisation) is **not yet approved**: do not implement it, but keep the code extensible for it (see §6).

**Purpose of this file:** tell whoever edits this repo (including Claude Code) exactly what to change so that the Aim 1 analyses can be run. Where this spec does not decide something, **keep the current behaviour of the codebase and make it a config option** (see §7).

---

## 1. Context

We are merging two master's-thesis studies into one paper and re-analysing both datasets with **one harmonised pipeline**. Earlier results from either thesis are not reused.

| | Cohort Gies | Cohort Lucas |
| --- | --- | --- |
| Participants | 14 (P1, P3–P15; P2 dropped out) | 13 |
| Protocols | Ramp, Easy (L→M), Medium (L→H), Hard (M→H); the three PRBS share one sequence at different intensities | Ramp, PRBS, Interval, Step |
| Ground truth | Cosmed Quark CPET, breath-by-breath V̇O₂ | same |
| Sensors | Monark LC7TT (work rate), Tymewear VitalPro (HR, V̇E, BR), PortaMon NIRS (O₂Hb, HHb, TSI), Sacoma scale (SMM) | same |

The codebase is Lucas's code, which was derived from Gies's original code ([public repo](https://github.com/lucasjansen024/AI-Thesis-Code)).

**Aim 1 question:** which input modalities, alone and in combination, give the best continuous V̇O₂ prediction? Models are trained on the **pooled** dataset. Within-cohort runs provide reference values, which Aim 2c will need later.

**Model:** TCN only. The GRU is removed from the main analysis (keep the code; don't run it).

---

## 2. Required pipeline changes

### 2.1 Scaling: no information from the test participant

- Fit **every** scaler (all input features **and** the V̇O₂ target) on the training participants of that fold only. Then apply it to the held-out participant.
- Inverse-transform predictions to mL·min⁻¹ before computing any metric.
- Keep the current transforms: work rate min–max; other continuous features z-score; sex coded 1/0 and not scaled.
- **Gies's original code** scaled work rate over all participants, so check that no global scaling remains anywhere.

### 2.2 Seeds: five runs, no selection

- Train every (configuration × fold) with **5 seeds** (e.g. 0–4). Seed Python, NumPy and TensorFlow; enable op determinism if feasible.
- Save every run. **Never select among seeds** (Gies's original code kept the best of 10 runs per test participant; that must not come back).
- Report each metric as mean ± SD across seeds. Also store the seed-averaged prediction per second for later analyses.

### 2.3 One training procedure for every run (fixed in advance, no tuning)

| Setting | Value |
| --- | --- |
| Architecture | TCN: 5 causal dilated Conv1D layers, kernel 7, 16 filters, dilations 1/2/4/8/16, residual blocks, layer norm, ReLU, dropout 0.2 |
| Receptive field | 187 s |
| Optimiser | Adam; learning rate 5e-5 by default, final value set by the training-settings pilot (§3.0) |
| Batch size | 32 |
| Epochs | 30 by default (Lucas's thesis); final value set by the training-settings pilot (§3.0), which may pick a cheaper setting |
| Loss | MSE |
| Checkpoint | Lowest **training** loss, as in Lucas's code |
| Validation set | None. Every model trains on all non-test participants |

**This exact procedure applies to every run that will be compared:** the pooled run, the within-cohort runs and, later, the Aim 2 models (including the external-cohort models in Aim 2c). Because every run follows the same rule, differences in error reflect the data, not the training procedure. Training-set sizes differ by design (pooled 26, Gies 13, Lucas 12, external 13–14 participants); report them alongside the results.

**Never** use test-participant data to choose an epoch, a checkpoint or any setting. That includes per participant, per configuration and globally.

### 2.4 Windows and evaluation segment (keep current behaviour)

- For target second *t*, the input window is *t* − 187 … *t* − 1 (causal).
- Targets come only from the exercise phase. The end of the warm-up is used as context only, and rest is excluded.

### 2.5 HRR definition (changed)

```
HRR(t) = (HR(t) − HRrest) / (HRpeak − HRrest)
```

- **HRrest:** mean HR during the pre-exercise rest period of the **same visit**. Do not take the minimum across visits.
- **HRpeak:** `208 − 0.7 × age` (Tanaka) for **every participant in both cohorts**. Do not use ramp-derived HRpeak, the Hard-visit ratio, or 220 − age.
- Age must therefore be available in the metadata for all 27 participants.

### 2.6 NIRS processing (changed)

- **Features:** TSI and Hb-diff (`O₂Hb − HHb`), **not normalised**. Remove both the occlusion-based min–max and the within-visit min–max.
- Downsample from 10 Hz to 1 Hz by averaging each second.
- The only scaling is the training-fold z-score from §2.1.
- **Check:** is the PortaMon zero set at the start of each recording? If not, Hb-diff has an arbitrary offset per visit. Report this before running, because it may require a baseline correction (see §7).

### 2.7 Breath-by-breath cleaning (unchanged, same for both cohorts)

1. Remove breaths more than 3 SD from the mean of the previous 5 breaths.
2. Apply a 5-breath median filter.
3. Linearly interpolate to 1 Hz.

This applies to V̇O₂, V̇E and BR.

### 2.8 Data inclusion and the shared sensor mask

- **Repeated visits (Gies):** P8 Medium and P9 Hard were repeated after a partial failure. Use the **complete** visit and drop the partial one.
- **Shared sensor mask (primary analysis):**
    - Use only visits where work rate, Tymewear **and** NIRS are all valid, so every configuration trains and is evaluated on **identical** data.
    - Apply the mask at window level: a target second counts only if its full 187 s input window is valid for every sensor. This handles P11 Hard, where NIRS is missing in the second half.
- **Expected Gies exclusions** (from the availability spreadsheet; verify against the raw files):

  | Problem | Visits |
  | --- | --- |
  | Tymewear missing | P6 Ramp, P7 Ramp, P7 Hard, P9 Ramp, P11 Ramp, P13 Ramp, P13 Easy, P14 Medium |
  | NIRS missing, whole visit | P7 Ramp, P7 Medium, P8 Easy, P11 Ramp, P11 Easy, P14 Medium, P15 Ramp |
  | NIRS missing, second half only | P7 Hard, P11 Hard |

  That leaves about 44 of 56 Gies visits. All 52 Lucas visits are expected to be valid; verify.
- **Data inventory output:** a CSV with participant, cohort, protocol, sensor, valid seconds, included (y/n) and reason. It will become the supplementary data-flow figure.
- **Optional sensitivity flag** (`mask: per_config`): each configuration uses all visits valid for its own sensors. Not the primary analysis.

### 2.9 IDs and grouping

- Give every sample a globally unique `participant_id` (e.g. `G01…`, `L01…`) and a `cohort` label.
- All splits are by participant: the held-out participant's data never appears in training.
- Use "sex", not "gender", in all code outputs and column names.

---

## 3. Aim 1 experiments

### 3.0 Training-settings pilot (run first; picks one global setting)

**Goal:** find the cheapest training setting that performs as well as the 30-epoch reference. Fewer epochs would cut Aim 1's compute dramatically. The pilot picks **one** setting, which then applies to every run (§2.3). It also measures time per training run.

**Arms (learning rate × epochs):**

| Arm | Learning rate | Epochs | Note |
| --- | --- | --- | --- |
| R | 5e-5 | 30 | Reference (Lucas's thesis) |
| A | 5e-5 | 10 | Same learning rate, fewer epochs |
| B | 2e-4 | 10 | Higher learning rate compensates for fewer updates |
| C | 5e-4 | 1 | Gies's original setting |
| D | 5e-4 | 3 | Between C and B |

For every arm, log training and validation loss per epoch. That lets arms R and A also show what 1, 3 and 5 epochs would give at the same learning rate, without extra runs.

**Data splits (validation only; no test participants are used):**

- **Pooled:** 3 different validation splits. Each holds out 3 participants (mixed cohorts) as validation and trains on the remaining 24. The test participants of the full runs play no special role here.
- **Within-cohort (essential):** 2 validation splits per cohort, each holding out 2 participants. One epoch over ~12 participants gives half the weight updates of one epoch over ~26, so a setting that is fine pooled can undertrain the smaller within-cohort models.
- **Configurations:** `WR`, `HR+VENT` and `WR+HR+VENT+TSI+HBDIFF+ANTHRO`.
- **Seeds:** 2.
- **Size:** 5 arms × 3 configurations × 2 seeds × 7 splits ≈ 210 trainings. Most arms are short, so this is cheap.

**Decision rule (write it down before running):**

- Choose the **cheapest arm** (fewest epochs) whose mean validation RMSE is within **10 mL·min⁻¹** of the best arm.
- That must hold separately in the pooled splits and in both within-cohort splits, for all three configurations.
- If no cheaper arm qualifies everywhere, keep the reference (R).
- **Never use test-participant results for this choice.** Gies's original 1 epoch was chosen from test performance, which is one of the faults being fixed.

**Methods statement:** "Training settings were selected once in a pilot on held-out validation participants and then fixed for all analyses." Note one limitation: pilot validation participants later appear as test participants in the full LOSO runs. Because only one global setting is chosen from a handful of options, the effect is minimal.
### 3.1 Input domains

Each domain has one fixed feature set: when a domain is used, all of its features are used (e.g. HR always comes with HRR). There are no sub-combinations within a domain. **The one exception is NIRS:** TSI and Hb-diff come from the same sensor but are treated as two completely separate domains.

| Domain | Sensor | Features |
| --- | --- | --- |
| `WR` | Bike (Monark) | work rate |
| `HR` | Tymewear VitalPro | HR, HRR |
| `VENT` | Tymewear VitalPro | V̇E, BR |
| `TSI` | PortaMon (NIRS) | TSI |
| `HBDIFF` | PortaMon (NIRS) | Hb-diff |
| `ANTHRO` | Sacoma scale | sex, SMM (constant within a visit, repeated at every time step) |

**Configurations:** every non-empty subset of the five time-varying domains {WR, HR, VENT, TSI, HBDIFF} (31), each with and without ANTHRO, gives **62 configurations**. ANTHRO on its own is excluded because it has no time course. TSI and HBDIFF may appear together in a configuration like any other two domains.

> If TSI and Hb-diff should never appear in the same configuration, the NIRS choice becomes {none, TSI, Hb-diff}. That gives 23 × 2 = 46 configurations. Make this a config flag (§7).

Generate the configuration list programmatically from the domain table and give each a stable, readable ID (e.g. `WR+HR+ANTHRO`).

### 3.2 Runs

| Run ID | Data | Cross-validation | Folds |
| --- | --- | --- | --- |
| `A1_pooled` | both cohorts, all protocols | leave-one-subject-out (LOSO) | 27 |
| `A1_gies` | Cohort Gies only | LOSO | 14 |
| `A1_lucas` | Cohort Lucas only | LOSO | 13 |

All three runs use LOSO and the single procedure in §2.3, so pooled and within-cohort results are directly comparable. Grouped k-fold and train/validation/test layouts were considered and deliberately left out: they would make the pooled run procedurally different from the within-cohort runs.

- **Size (62 configurations × 5 seeds):**

  | Run | Trainings |
  | --- | --- |
  | `A1_pooled` | 8,370 |
  | `A1_gies` + `A1_lucas` | 8,370 |
  | **Total** | **16,740** |

  If this is too much, run the within-cohort runs only for the configurations carried forward to Aim 2 (saves ~8,000). Do not switch to a different fold scheme.
- **Priority:** run the pooled runs first. The within-cohort runs are Aim 2c's reference. Ideally run all configurations; at minimum run the ones carried forward to Aim 2.
- The code should accept subsets of configuration IDs, folds and seeds from the command line, so runs can be parallelised (job arrays). Make it resumable by skipping runs whose outputs already exist.

### 3.3 Outputs per training run

- **Predictions** (parquet): `run_id, config_id, seed, fold, participant_id, cohort, protocol, visit_id, t_s, work_rate, vo2_true, vo2_pred`.
- **Per-visit metrics:** RMSE, MAE, bias and number of seconds for each participant × visit × seed.
- **Manifest:** git commit, full resolved config, seed, package versions, start/end time.

### 3.4 Evaluation and reporting

- **Primary metric:** RMSE (mL·min⁻¹) per participant-visit. Per configuration, report the mean ± SD across participants (seed-averaged) and the SD across seeds.
- **Secondary metrics:**
    - repeated-measures correlation (`pingouin.rm_corr`)
    - Bland–Altman bias and 95% limits of agreement, using the repeated-measures method (Bland & Altman, 2007)
- **Breakdowns of the pooled run:** by cohort and by protocol.
- **Planned statistics** (can be scripted after the runs):
    - linear mixed model `log(RMSE) ~ config + protocol + cohort + (1 | participant)`
    - contrasts against `WR` alone, Holm-corrected
    - equivalence tests (TOST) against `WR` alone, with the margin set before looking at the results (default ±25 mL·min⁻¹; §7)
- **No winner selection inside the pipeline.** Configurations are chosen for Aim 2 afterwards, using a rule written down beforehand. For example: the lowest pooled mean RMSE, but if a simpler configuration is within the equivalence margin, choose the simpler one.

---

## 4. Suggested implementation order

1. Read the repo. Write a short `MAPPING.md` that links each item in §2 to the file and function where it lands, and lists anything already compliant.
2. Load the Gies data into the same format as Lucas's (protocol names Ramp/Easy/Medium/Hard), apply the participant-ID scheme and write the data inventory script (§2.8).
3. Make the preprocessing changes: HRR (§2.5), NIRS (§2.6), the shared mask (§2.8).
4. Set up a config file (YAML) defining domains, configurations, runs, seeds and the open-decision flags from §7.
5. Training loop: seeds, the fixed procedure from §2.3, saved predictions and manifests, resumable runs, training (and, in the pilot, validation) loss logged per epoch.
6. Evaluation script: metric tables and breakdowns.
7. Add the tests in §5, then a smoke run (2 configurations × 2 folds × 1 seed).
8. Run the training-settings pilot (§3.0); record the chosen setting and the time per run.
9. Run the full Aim 1 experiments with that setting.

---

## 5. Acceptance checks (automated tests where possible)

- **No leakage:** the data used to fit any scaler contains no rows from the test participant.
- **Fold integrity:** the test participant is absent from training. Pooled LOSO has exactly 27 folds, and the within-cohort runs have 14 and 13, each with a unique test participant.
- **One procedure:** all runs use the identical training settings from §2.3; only the training data differ. Assert that the resolved hyperparameters are the same across runs in the manifests.
- **Identical data across configurations:** for a given fold, the number of evaluated seconds (and their IDs) is identical for all 62 configurations.
- **Configuration list:** 62 unique configurations; each domain's features always appear together (e.g. never HR without HRR); TSI and Hb-diff can appear independently.
- **HRR:** HRpeak equals `208 − 0.7 × age`, and HRrest comes from the same visit's rest period. Spot-check values (expect roughly −0.1 to 1.1).
- **Causality:** the last input sample of each window is *t* − 1.
- **Seeds:** different seeds give different models; rerunning the same seed reproduces the result within tolerance.
- **Inventory:** excluded visits match §2.8 (or the differences are explained).

---

## 6. Out of scope now: Aim 2 (keep the code extensible)

Aim 2 (pending approval) will test generalisation:

- **2a, unseen intensity (Gies):** for each target intensity, compare training with the target included against training with it excluded. The included set is randomly subsampled to exactly the size of the excluded set: `fraction = (total − target) / total`.
- **2b, unseen protocol (Lucas):** the same design across Ramp, PRBS, Interval and Step.
- **2c, external cohort:** train on all of one cohort and test on the other, compared with that cohort's within-cohort LOSO from Aim 1. The external models use exactly the same training procedure (§2.3).
- **Supporting analysis:** the temporal structure of residuals (e.g. runs tests or run lengths).

**What this means for the code now:**

- Define a training set as an explicit list of (participant, visit) pairs, plus an optional seeded subsampling fraction.
- Keep protocol labels on every saved prediction.
- Keep the run types (`run_id`) generic.

---

## 7. Open decisions: use these defaults and make each one a config flag

| Decision | Default until decided | Note |
| --- | --- | --- |
| TSI and Hb-diff in the same configuration | allowed (62 configurations) | Alternative: mutually exclusive (46 configurations) |
| Hb-diff channel | current code (mean of the 3 channels) | Gies used the 40 mm channel only |
| Hb-diff offset / baseline | raw as exported | Check the PortaMon zero point (§2.6); fallback: subtract the pre-exercise rest mean |
| HRrest | mean of the same visit's pre-exercise rest | Proposal says "resting period before exercise" |
| SMM | current code behaviour (check: per-visit value or participant mean) | SMM was measured before every visit; a participant mean reduces BIA noise |
| NIRS fit-factor threshold | current code (99.9%) | Gies used 99%; affects Gies exclusions |
| Training setting (learning rate × epochs) | chosen by the pilot rule (§3.0); reference 5e-5 × 30 | One global setting for all runs; the pilot uses validation participants only |
| Pilot tolerance | within 10 mL·min⁻¹ of the best arm's validation RMSE | Fix before running the pilot |
| Equivalence margin | ±25 mL·min⁻¹ | Must be fixed before the results are seen |
| Within-cohort runs | all 62 configurations | Minimum: the configurations carried forward (saves ~8,000 trainings) |
