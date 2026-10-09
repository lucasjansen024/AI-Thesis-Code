# VO₂ estimation from wearable sensors

Two master's-thesis datasets are re-analysed with one harmonised pipeline, to estimate second-by-second pulmonary oxygen uptake (V̇O₂) from wearable and ergometer signals with a temporal convolutional network.

## Language

### Data

**Cohort**:
One of the two studies whose data are pooled: **Gies** (2024, 14 participants) or **Lucas** (2026, 13 participants).
_Avoid_: dataset, study (when you mean a cohort)

**Participant**:
A person who completed a cohort's visits, identified by a **participant_id** that encodes the cohort and the original study number: `G01, G03–G15` (Gies) and `L01–L10, L12–L14` (Lucas).
_Avoid_: subject, bare study labels such as "P12"

**Protocol**:
The prescribed work-rate pattern of a visit. Gies: Ramp, Easy (low↔moderate PRBS), Medium (low↔heavy PRBS), Hard (moderate↔heavy PRBS). Lucas: Ramp, PRBS, Interval, Step.
_Avoid_: test type

**Visit**:
One recording session in which one **Participant** performs one **Protocol**, identified by a **visit_id** (`G08_Medium`). The raw folder name it came from is its **source_visit** (`Medium2`).
_Avoid_: session, measurement, trial

**Repeated visit**:
A visit performed again after a partial failure (Gies P8 Medium, P9 Hard). Only the complete one is used.

**Failed visit**:
An aborted visit that was performed again later (Gies P13 FailedRamp). Never used.

**Repetition**:
One of the two identical PRBS sequences within a PRBS visit (Gies Easy/Medium/Hard, Lucas PRBS). It is the **Unit** of validity in those visits.

**Phase**:
A segment of a visit's timeline: **rest** (before exercise, excluded), **warm-up** (its end is used only as input context, and its data quality never makes a unit invalid) and **exercise** (the only source of prediction targets).

**Sensor**:
A measuring device: metabolic cart (ground-truth V̇O₂, plus a chest-strap HR used as backup), Tymewear strap (HR, V̇E, BR), PortaMon NIRS (haemoglobin signals, TSI), ergometer (work rate), BIA scale (SMM).

**Ideal work rate**:
The work-rate sequence reconstructed from the protocol definition and the participant's prescribed power levels. Used instead of the measured power.
_Avoid_: measured power, Power column

**Fit factor**:
The PortaMon's quality indicator (%) for how well its three source–detector distances agree when computing TSI.
_Avoid_: fit file (the scale's Fitdays export is unrelated)

**Annotation**:
A per-visit or per-repetition data-quality label in a cohort's config (e.g. "NIRS valid in repetition one only", "no HR"). It is how data were originally selected. It no longer decides validity: the automatic checks and the **NIRS review** replace it.

**Unit**:
The span over which validity is decided: a **Repetition** in PRBS visits, otherwise the visit's whole exercise phase. A **Sensor group** is valid or invalid for a whole unit, never for single seconds.
_Avoid_: valid second

**Sensor group**:
Inputs whose validity is judged together: target V̇O₂, work rate, HR (from any HR sensor), ventilation (Tymewear V̇E and BR), NIRS (TSI and Hb-diff), and static (sex, SMM).

**NIRS review**:
The developer's visual judgement of NIRS quality per visit or repetition. It can exclude units that pass the automatic checks.

**Shared mask**:
The rule that a target second is used only if its **Unit** is valid for every **Sensor group**. The seconds before it in its **Window** are input context: they are admitted whatever their own validity, as long as their data exists. It makes all configurations train and test on identical data.

**Prepared data**:
Per-visit, per-sensor signals after file selection, synchronisation and trimming, still on each sensor's native time base.

**Processed data**:
A visit's merged 1 Hz table of features and target, with validity information, before scaling.

### Model inputs

**Domain**:
One of the five fixed input groups: `WR` (work rate), `HR` (HR, HRR), `VENT` (V̇E, BR), `NIRS` and `ANTHRO` (sex, SMM). A domain's features are always used together, except that `NIRS` enters through exactly one **NIRS variable**.
_Avoid_: feature group, signal group

**NIRS variable**:
One of the two alternative features of the `NIRS` domain: `TSI` or `HBDIFF` (Hb-diff). A configuration uses at most one, never both.

**Configuration**:
A non-empty set of time-varying **Domains** (`NIRS` as one **NIRS variable**), optionally plus `ANTHRO`. It is named by joining the IDs, with NIRS written as its variable (`WR+HR+ANTHRO`, `HR+TSI`). There are 46.
_Avoid_: processor, input combination, model type

**HRR**:
Heart-rate reserve, `(HR − HRrest) / (HRpeak − HRrest)`. HRrest is the mean HR in the same visit's rest phase; HRpeak is `208 − 0.7 × age`.

**Hb-diff**:
Oxygenated minus deoxygenated haemoglobin from the NIRS transmitter channel Tx3, as exported: not normalised and not baseline-corrected.
_Avoid_: HbDiff (the Lucas device's own export column, which is not used)

**SMM**:
Skeletal muscle mass in kg: the scale's skeletal-muscle percentage × body weight, measured before each visit.
_Avoid_: muscle mass (the scale's separate, larger "Muscle mass" value)

### Training and evaluation

**Window**:
The 187 input seconds `t−187 … t−1` used to predict V̇O₂ at target second `t`.

**Fold**:
One leave-one-subject-out split: one test **Participant**; every other participant in the run trains the model.

**Run**:
The fixed set of participants and folds being evaluated: `A1_pooled` (both cohorts), `A1_gies`, `A1_lucas`.
_Avoid_: experiment (for a single training)

**Training**:
One model fitted for one (**Run**, **Configuration**, **Fold**, **Seed**) combination.

**Seed**:
The random seed of a **Training**. Every combination is trained with five seeds and none is ever selected over another. A seed fixes the model's initial weights and its shuffle order (one permutation per epoch, shared by all configurations with that seed); dropout draws belong to the packed job.

**Checkpoint**:
The epoch whose average training loss (dropout active) is lowest; that epoch's weights are the trained model. Never chosen on validation or test data.

**Pilot**:
The training-settings pilot that picks one global learning-rate × epochs setting. Its candidate settings are **arms** (R, A, B, C, D). It is LOSO, like the **Runs**: one split per held-out **Participant** in each of the pooled, Gies and Lucas families (27 + 14 + 13). The dev chooses the setting from its table.

**Trajectory**:
One pilot training at a fixed learning rate, logged every epoch. An arm with fewer epochs at the same learning rate is read off a longer trajectory (A from R, C from D), since its training is identical up to that epoch.

**Smoke run**:
A tiny **Run** (2 **Configurations** × 2 **Folds** × 1 **Seed**, 3 epochs) that checks the pipeline end to end before the **Pilot**. It is a plumbing check, its results are not used, and it is not the **Pilot**.

**Pilot smoke**:
A tiny **Pilot** slice (1 split × 1 **Trajectory** × 2 **Configurations** × 1 **Seed**) that checks the pilot-only code path (held-out logging). It picks no setting.

**Seed-averaged metric**:
The mean of a metric over the five **Seeds**, each seed's metric computed from its own predictions. It estimates how one **Training** performs.
_Avoid_: ensemble RMSE (the metric of the averaged prediction, which is lower)

**Seed-averaged prediction**:
The per-second mean of the five **Seeds**' predictions. It is stored for later analyses (plots, residual structure), not used to compute the reported metrics.

**VO₂peak**:
The highest 30-s rolling mean of V̇O₂ over a ramp visit's evaluated seconds, for measured and predicted V̇O₂ alike.
_Avoid_: VO₂max (no plateau criterion is checked)

## Relationships

- A **Cohort** has many **Participants**. A **Participant** has one **Visit** per **Protocol** (Gies P14 has no Medium visit).
- A **Configuration** contains one or more **Domains**. `ANTHRO` never appears alone. `TSI` and `HBDIFF` never appear together.
- A **Run** has one **Fold** per participant (27, 14 or 13). Each fold trains every **Configuration** × **Seed**.
- A **Visit** has one **Unit** (Ramp, Step, Interval) or two (PRBS visits: one per **Repetition**). Each **Unit** is valid or invalid per **Sensor group**.
- The **Shared mask** is computed from **Unit** validity across all **Sensor groups**; every **Configuration** in a **Fold** sees the same masked data.

## Example dialogue

> **Dev:** "G04's metabolic cart stopped halfway through the Hard visit. Do we drop the visit?"
> **Domain expert:** "No. Target V̇O₂ is a **sensor group** too, so repetition two is an invalid **unit** and repetition one stays valid. The **shared mask** keeps every target second whose whole **window** lies in valid units, for every **configuration**."
> **Dev:** "And G08 has two Medium visits?"
> **Domain expert:** "That's a **repeated visit**. `visit_id` G08_Medium refers to the complete one, whose **source_visit** is Medium2."

## Flagged ambiguities

- "Domain": the legacy code groups inputs into five values (VE, HR, NIRS, WR, Fit). **Domain** here means only the five input domains above.
- The handoff (§3.1) treats TSI and Hb-diff as two separate domains that may be combined (62 configurations). Here they are **NIRS variables** of one domain (46 configurations).
- "P12", "P13", "P14" exist in both cohorts. Always use a **participant_id**.
- "Lucas data" can mean the Lucas **Cohort** (2026) or an unrelated 2025 dataset found in the archives (P01–P10, LP01–LP10). That 2025 dataset is excluded; **Cohort** Lucas means 2026 only.
- "gender" → **sex** in every output and column name.
- The handoff's pilot configurations (§3.0: `WR`, `HR+VENT`, `WR+HR+VENT+TSI+HBDIFF+ANTHRO`) are replaced by five: `WR`, and with and without `WR` the full set for each **NIRS variable**. The handoff's validation splits are replaced by LOSO.
- "V̇E" and "BR" can come from the metabolic cart or from the Tymewear strap. Only the Tymewear signals are model inputs; the cart's V̇E shares the flow measurement V̇O₂ is computed from.
