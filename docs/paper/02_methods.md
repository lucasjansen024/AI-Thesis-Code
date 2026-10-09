# Methods (draft v0.1, 9 Oct 2026)

> **Notes for the authors (delete before submission)**
> - Describes the pipeline as implemented at commit `7bd3739`; the source of truth is `docs/METHODOLOGY.md`. When that file changes, update this one.
> - Deliberately elaborate. Candidates to move to supplementary material: §2.5.1 (file selection and synchronisation), the threshold rationales in §2.6, the packed-training paragraph in §2.9, and Table S1 (all 46 configurations).
> - **[Bold square brackets]** = information missing or decisions pending (see `METHODOLOGY.md` §15).
> - Citations: `[Author Year]`, listed in `docs/paper/references.md`.
> - Terminology used throughout: *visit* (one protocol performed by one participant), *repetition* (one of the two identical PRBS sequences in a visit), *segment* (the unit of quality assessment: a repetition, or the whole exercise phase), *domain* (input group), *configuration* (set of domains).

---

## 2.1 Participants

Data from two independent studies, conducted with the same core equipment at **[laboratory — confirm: Human Performance Laboratory, Vrije Universiteit Amsterdam, for both cohorts]**, were pooled (Table 1).

**Cohort G** (2024) included 14 healthy, recreationally active adults (8 men, 6 women; one further participant withdrew and is not included). **Cohort L** (2026) included 13 healthy adults (8 men, 5 women) without chronic conditions likely to alter the normal response to exercise. Participants were identified by cohort and original study number (G01, G03–G15; L01–L10, L12–L14).

All participants gave written informed consent and were informed of their right to withdraw without consequence. Cohort G was approved by the Natural Sciences & Engineering Sciences (NES) ethics committee **[institution]** (request #240302). Cohort L was approved by the Scientific and Ethical Review Board (VCWE) of the Faculty of Behavioural and Movement Sciences, Vrije Universiteit Amsterdam (VCWE-2026-113). Both studies conformed to the Declaration of Helsinki **[confirm]**.

**Table 1.** Participant characteristics. **[Recompute from the harmonised data: age and sex from the CPET headers, height, body mass and skeletal muscle mass from the first visit, peak V̇O₂ and peak power from the ramp with the 30-s rule of §2.12. Values below are from the original reports and are provisional.]**

| | Cohort G (n = 14) | Cohort L (n = 13) | Pooled (n = 27) |
|---|---|---|---|
| Sex (M/F) | 8/6 | 8/5 | 16/11 |
| Age (y) | 25.8 ± 3.2 | 25.0 ± 2.6 | **[ ]** |
| Height (cm) | 180 ± 9 | 178.1 ± 9.8 | **[ ]** |
| Body mass (kg) | 74.2 ± 9.5 | 70.1 ± 13.5 | **[ ]** |
| Skeletal muscle mass (kg) | **[ ]** | **[ ]** | **[ ]** |
| Peak V̇O₂ (L·min⁻¹) | 3.47 ± 0.62 | 3.14 ± 0.83 | **[ ]** |
| Peak V̇O₂ (mL·kg⁻¹·min⁻¹) | 46.9 ± 7.1 | 44.1 ± 7.3 | **[ ]** |
| Peak power (W) | 348 ± 62 | 297 ± 86 | **[ ]** |
| GET (L·min⁻¹) | 2.22 ± 0.37 | 2.10 ± 0.47 | **[ ]** |

Values are mean ± SD. GET, gas exchange threshold.

## 2.2 Experimental design

Each participant completed four visits on separate days, all on the same electronically braked cycle ergometer (Monark LC7TT, Monark Exercise AB, Sweden **[city: Lucas thesis says Varberg; confirm]**). The first visit was always a ramp-incremental test, from which the gas exchange threshold (GET) and peak values were determined to prescribe the intensities of the subsequent visits. The remaining three visits were performed in random order. In cohort G, these were three PRBS visits with different intensity combinations. In cohort L, these were a PRBS visit, an interval visit and a step visit (§2.3). **[Minimum time between visits for each cohort.]** Participants were asked to avoid strenuous exercise and alcohol for 24 h, caffeine on the day of testing and food for 2 h before each visit **[confirm that this applied to cohort G as well]**. Tests were performed in a temperature-controlled laboratory (18–21 °C, 50 ± 10 % relative humidity **[confirm for cohort L]**). Participants pedalled at a self-selected cadence between 70 and 90 rpm, which was maintained across visits **[cohort G: the self-selected cadence was reproduced in each PRBS visit; confirm cohort L]**.

Every visit began with 2 min of quiet seated rest on the ergometer with all sensors attached, followed by a warm-up and the exercise phase. The rest phase served only to determine resting heart rate (§2.5.6). The end of the warm-up served as input history for the models but was never used as a prediction target. Only the exercise phase supplied prediction targets.

## 2.3 Exercise protocols

The protocols are summarised in Table 2 and shown in Figure 1 **[Figure 1: work-rate profiles of all seven protocols for a representative participant per cohort; adapt Figure 1 of the Gies draft and of Lucas's thesis]**. Intensities between GET and peak are expressed as Δx % = GET + x % × (peak − GET), in power.

**Ramp-incremental test (both cohorts).** After 2 min of rest, participants cycled for 4 min at 30 W, after which work rate increased linearly by 20, 25, 30 or 35 W·min⁻¹ until task failure. The ramp rate was chosen per participant from body size and self-reported activity level, aiming for 8–12 min of ramp exercise. Task failure was defined as a cadence below 60 rpm despite strong verbal encouragement **[cohort L: for more than 5 s]**. The ergometer implemented the ramp as small steps (1 W every 3 s at 20 W·min⁻¹; 5 W every 12 s at 25 W·min⁻¹; 1 W every 2 s at 30 W·min⁻¹; 7 W every 12 s at 35 W·min⁻¹; one participant's 20 W·min⁻¹ ramp used 4 W every 12 s). The GET was determined by visual inspection of the standard gas-exchange and ventilatory indices: a disproportionate increase in V̇CO₂ relative to V̇O₂, an increase in the ventilatory equivalent for O₂ without an increase in the ventilatory equivalent for CO₂, and an increase in end-tidal PO₂ without a fall in end-tidal PCO₂ [Goulding 2017; Wasserman 1984]. The corresponding power was corrected for the mean response time of V̇O₂ **[confirm the GET method and the mean-response-time correction for cohort L]**.

**PRBS visits (cohort G).** Work rate switched between two levels according to a pseudorandom binary sequence with 30-s units. After 2 min of rest, a 3.5-min PRBS warm-up at the visit's two work rates was followed by two identical repetitions of a 16-min sequence (32 units). The three visits used different level pairs, chosen to keep the response within the moderate domain, to span the moderate and heavy domains, or to stay mostly within the heavy domain: *Easy* (L→M), 25 W ↔ 90 % GET; *Medium* (L→H), 25 W ↔ Δ30 %; and *Hard* (M→H), **[GET or 90 % GET — confirm from protocol files]** ↔ Δ30 %.

**PRBS visit (cohort L).** After 2 min of rest and a 3.5-min warm-up at 80 % GET, work rate switched between 25 W and Δ40 % according to a PRBS with 30-s units generated by a linear-feedback shift register, with runs lasting 30 s to 2.5 min. The 15.5-min sequence (31 units) was performed twice (31 min).

**Interval visit (cohort L).** After rest and the same warm-up, participants performed 10 cycles of 2 min at 25 W and 1 min at Δ70 %, followed by 1 min at 25 W (31 min).

**Step visit (cohort L).** After rest and the same warm-up, work rate followed 6 min at 80 % GET, 6 min at Δ30 %, 6 min at 80 % GET, 3 min at Δ60 %, 5 min at 25 W and 5 min at 60 % GET (31 min).

**Table 2.** Protocol timelines as used in the analysis. Times are from the start of the recording.

| Cohort | Protocol | Rest | Warm-up | Exercise phase (targets) | Quality segments |
|---|---|---|---|---|---|
| G, L | Ramp | 0–120 s | 120–360 s, 30 W | 360 s → task failure | 1 |
| G | Easy, Medium, Hard (PRBS) | 0–120 s | 120–330 s, PRBS | 330–2250 s (2 × 960 s) | 2 (repetitions) |
| L | PRBS | 0–120 s | 120–330 s, 80 % GET | 330–2190 s (2 × 930 s) | 2 (repetitions) |
| L | Interval, Step | 0–120 s | 120–330 s, 80 % GET | 330–2190 s | 1 |

Two cohort G visits were repeated after a partial failure (one Medium, one Hard); only the complete repetition was used. One ramp test was aborted and repeated; only the repeat was used. One participant of cohort G did not perform the Medium visit. This gives 55 candidate visits in cohort G and 52 in cohort L.

## 2.4 Measurements

**Pulmonary gas exchange (ground truth).** Gas exchange and ventilation were measured breath-by-breath with a metabolic cart (Quark CPET, COSMED, Rome, Italy). Before each test, the gas analysers were calibrated with gases of known concentration and the turbine with a 3-L syringe (Hans Rudolph, Kansas City, MO, USA), according to the manufacturer's instructions. The cart also recorded HR from a chest strap **[model]**, which served only as a backup HR source (§2.5.6). The cart's ventilation and breathing frequency were never used as model inputs, because they are derived from the same flow signal as V̇O₂.

**Chest strap.** HR, V̇E and BR were recorded with a Tymewear VitalPro strap (Tymewear, Boston, MA, USA), which combines an HR sensor with a respiratory sensor that estimates ventilation from chest-wall movement. The strap was placed around the thorax directly below the pectoral muscles (men) or below the breasts (women) and calibrated in the companion app with three full inhalation–exhalation cycles. No calibration against the metabolic cart was performed.

**Near-infrared spectroscopy.** Oxygenation of the right vastus lateralis was measured with a continuous-wave, spatially resolved NIRS device (PortaMon, Artinis Medical Systems, **[Elst or Zetten — confirm]**, the Netherlands; cohort L used the PortaMon MKIII). The device has three light sources at 30, 35 and 40 mm from the detector and emits at ~760 and ~850 nm. It reports relative changes in oxygenated (O₂Hb), deoxygenated (HHb) and total haemoglobin per source, and the tissue saturation index (TSI, %) from the spatially resolved signal together with a fit factor that indicates how well the three distances agree. The probe was placed midway along the muscle belly **[cohort L: midway between the greater trochanter and the lateral femoral epicondyle]** and fixed with tape and bandages **[cohort L: plus a blackout sleeve]** to limit movement and ambient light. All recordings were exported at 10 Hz. An arterial occlusion was performed after exercise for NIRS calibration **[confirm protocol for both cohorts]**; these data were not used (§2.5.7).

**Body composition.** Before each visit, body mass and skeletal-muscle percentage were measured with a bioelectrical-impedance smart scale (Sacoma, Amsterdam, the Netherlands; Fitdays app). Age and sex were taken from the participant profile in the metabolic-cart software.

**Work rate.** The ergometer executed each submaximal protocol from a stored protocol file. The ramp was executed from the cart's protocol settings.

## 2.5 Data processing

Both cohorts were processed from the raw exports with one pipeline (Python; §2.15). Results from the original studies were not reused.

### 2.5.1 File selection and synchronisation

Files were selected by explicit rules, recorded per visit. For gas exchange, only breath-by-breath exports were used; time-averaged (10-s) exports were never used. The metabolic cart's clock served as the reference time base. Chest-strap data were already on this clock in cohort L. In cohort G, they were shifted by a per-visit offset determined during the original study by aligning the strap's and the cart's ventilation. NIRS recordings were aligned using the event marker set at a known protocol time during the recording. When a file contained several markers (8 files, from restarted recordings), the last one was used. No manual per-visit corrections were applied.

### 2.5.2 Time base and phases

Each visit was represented on a 1-Hz grid from the start of the rest phase to the end of exercise. Phases were assigned from the nominal protocol timeline (Table 2), not from manual phase marks. The end of a ramp test was defined as the last breath with a cadence of at least 60 rpm.

### 2.5.3 Ground truth: V̇O₂

Breath-by-breath V̇O₂ was cleaned in three steps, identical for both cohorts. (1) Each breath that differed by 3 SD or more from the mean of the preceding five breaths was removed. (2) A centred 5-breath median filter was applied. (3) Values were linearly interpolated to the 1-Hz grid. Interpolation was never trusted across long gaps: a second was marked as missing when the raw breaths on either side of it were more than 20 s apart.

### 2.5.4 Work rate

The work-rate input was the work rate prescribed by the protocol, reconstructed second by second from the ergometer's protocol file (submaximal visits) or from the ramp settings (ramps). We did not use the power recorded by the cart. It is logged per breath, so it is irregularly sampled and inconsistent with the actual load changes. The reconstruction matched the recorded power closely (median absolute difference 0–1 W over the exercise phase of every ramp). For the two repeated cohort G visits, which lacked their own protocol file, the file of the original attempt (same protocol) was used. For one cohort L PRBS visit without a protocol file, the timing was taken from another participant's identical PRBS file, and the high and warm-up levels from the median recorded power during those phases.

### 2.5.5 Ventilation

V̇E and BR were taken from the strap's breath-by-breath output and cleaned with the same three steps as V̇O₂, except that the median filter was trailing (current and four preceding breaths), so that inputs depend only on past breaths. A second was marked as missing when raw breaths were more than 20 s apart. Ventilation counted as available only when both V̇E and BR were available.

### 2.5.6 Heart rate and heart-rate reserve

HR was taken from the chest strap when it passed the data-quality check described in §2.6. Otherwise, the cart's chest-strap HR was used when it passed the same check (5 visits). HR was linearly interpolated to 1 Hz without smoothing. Heart-rate reserve was computed per second as

HRR(t) = (HR(t) − HR_rest) / (HR_peak − HR_rest),

where HR_rest is the mean HR during the first 120 s (rest) of the same visit and from the same source, and HR_peak = 208 − 0.7 × age [Tanaka 2001] for every participant. Using the same visit's rest and an age-predicted peak avoids using information from other visits or from the test participant's maximal test. For one visit whose cart profile contained another participant's details, age was taken from the same participant's previous visit.

### 2.5.7 Muscle oxygenation

Two NIRS variables were derived: TSI, and the haemoglobin difference (Hb-diff = O₂Hb − HHb) of the source **[intended to be the 40-mm source; mapping of channel to distance to be verified]**. Both were used as exported, without normalisation to occlusion or to the visit's own range and without baseline subtraction. The only scaling was the fold-wise standardisation in §2.9. The zero point of the relative haemoglobin signals is arbitrary per recording. Subtracting the resting mean was examined visually and made little difference relative to the exercise-induced changes. All-zero rows and a non-physiological first row present in some exports were removed. The 10-Hz signals were averaged to 1 Hz (mean of the samples within each second). A second without NIRS samples was marked as missing. The post-exercise occlusion data lie outside the analysed time window, and the fit factor was used only for quality control (§2.6).

### 2.5.8 Anthropometrics

Sex was coded 1 for male and 0 for female. Skeletal muscle mass (SMM, kg) was computed as the scale's skeletal-muscle percentage multiplied by body mass. It was taken from the measurement on the day of the visit and was therefore constant within a visit. For one visit without a same-day measurement, the nearest earlier measurement (6 days before) was used. Note that SMM differs from the scale's separate "muscle mass" output. Sex and SMM were repeated at every second of the visit as inputs.

## 2.6 Data quality and inclusion

Data quality was assessed **before any model was trained**, with explicit rules for each sensor and each segment of the exercise phase. The verdicts were stored in a single table that the training code reads but never modifies. The aim was to remove recordings that were missing or physiologically implausible, not to remove difficult data, so no rule refers to prediction error.

**Segments and sensor groups.** Validity was decided per *segment*: each PRBS repetition separately, and otherwise the whole exercise phase of the visit (161 segments). Within a segment, it was decided per *sensor group*: V̇O₂ (target), work rate, HR, ventilation, NIRS and static (sex and SMM). A group was either valid or invalid for a whole segment, never for single seconds, because a corrupted second would also enter the input history of the following 187 s. Only exercise seconds were assessed. The quality of the warm-up, which serves as input history only, never invalidated a segment.

**Automatic criteria.**
- *Presence and gaps (all groups).* A group was invalid if its recording was missing or if the longest gap without samples within the segment exceeded 20 s.
- *Work rate.* Valid if the protocol definition was available (three visits used the substitutes described in §2.5.4).
- *Heart rate.* A source (strap first, then the cart's chest strap) had to show an HR range (95th − 5th percentile) of at least 20 bpm and a Pearson correlation of at least 0.5 with 30-s smoothed V̇O₂, both computed from the start of rest to the end of exercise, and HR data for at least 80 % of the rest phase (needed for HR_rest). These criteria detect sensors that record a constant or unrelated signal, which a coverage check alone misses. They are computed from rest onwards because during heavy PRBS exercise, genuine HR varies too little for an exercise-only check. They reproduced the original study's manual HR-source annotations exactly.
- *NIRS.* In addition to the gap rule, at least 95 % of the segment's NIRS samples had to have a fit factor of at least 99 %. No validated threshold exists for this index. A stricter per-sample threshold (99.9 %) would have excluded nine cohort L visits whose signals were otherwise of good quality. On the present data, this criterion excluded no segment.

**Visual NIRS review.** NIRS signals were additionally reviewed visually by one investigator, blinded to model results (no models had been trained). A review could exclude a segment that passed the automatic checks, but could not include a segment that failed them. Four segments were excluded on review: both repetitions of one visit, the second repetition of another, and the first repetition of a third, whose warm-up contained a TSI artefact that would have entered the input history of that repetition's first targets.

**Resulting dataset.** All 52 cohort L visits and 47 of 55 cohort G visits were included (Table 3; **[Figure S1: data-flow diagram]**). Exclusions were due to missing chest-strap recordings (4 ramps), unusable HR from both sources (2 visits) or from the only available source (1 visit), and unusable NIRS (1 visit). Two further cohort G visits contributed one repetition each: in one, the gas-exchange recording stopped during the second repetition; in the other, the first repetition was excluded on NIRS review. HR came from the cart's chest strap in 5 visits.

**Table 3.** Included visits per protocol (shared mask). **[Add target seconds per protocol from `summary/targets.csv`.]**

| Cohort | Ramp | Easy | Medium | Hard | PRBS | Interval | Step | Total |
|---|---|---|---|---|---|---|---|---|
| G | 9/14 | 12/14 | 13/13 ¹ | 13/14 ² | — | — | — | 47/55 |
| L | 13/13 | — | — | — | 13/13 | 13/13 | 13/13 | 52/52 |

¹ One visit with repetition 2 only. ² One visit with repetition 1 only.

**Prediction targets and shared mask.** A second t was used as a prediction target only if (i) it lay in the exercise phase, (ii) its segment was valid for **all six** sensor groups, (iii) V̇O₂ was available at t, and (iv) every input group had data (no gap > 20 s) throughout the preceding 187 s, the model's input window. The input window could extend into the warm-up or into a neighbouring repetition that was invalid for another reason, as long as the data were present. Requiring all groups for every configuration (the *shared mask*) means that all 46 input configurations were trained and evaluated on exactly the same seconds, so differences between configurations cannot arise from differences in the data. In total, **[N]** target seconds were available (**[N_G]** in cohort G, **[N_L]** in cohort L).

## 2.7 Input domains and configurations

Inputs were grouped into five domains, each corresponding to one sensor or physiological system (Table 4). Within a domain, features were always used together (e.g. HR always with HRR). The NIRS domain entered through exactly one of its two variables, TSI or Hb-diff, because both derive from the same measurement. The anthropometric domain has no time course and was only used in combination with at least one time-varying domain.

**Table 4.** Input domains.

| Domain | Features | Sensor | Scaling |
|---|---|---|---|
| WR | work rate | ergometer (protocol) | min–max |
| HR | HR, HRR | chest strap (cart chest strap as backup) | z-score |
| VENT | V̇E, BR | chest strap (respiratory sensor) | z-score |
| NIRS | TSI *or* Hb-diff | NIRS, vastus lateralis | z-score |
| ANTHRO | sex, SMM | cart profile, BIA scale | sex: none; SMM: z-score |

All combinations were evaluated. These are the 23 non-empty combinations of WR, HR, VENT and at most one NIRS variable, each with and without ANTHRO, giving **46 configurations** (Table S1), with 1 to 8 input features. Configurations are named by their domains, with NIRS written as its variable (e.g. `WR+HR+TSI+ANTHRO`).

## 2.8 Model

We used a causal temporal convolutional network [Bai 2018] with the architecture introduced for V̇O₂ prediction by Amelard et al. [Amelard 2021] and used in both preliminary studies (Figure 2 **[schematic; adapt Lucas's thesis Figure 2]**). The model maps the preceding 187 s of inputs (seconds t − 187 to t − 1) to V̇O₂ at second t. Input at t itself is not used, so the model is causal and could run in real time.

The network consists of five causal dilated one-dimensional convolutions (kernel size 7, 16 filters, dilation factors 1, 2, 4, 8 and 16), arranged in two residual blocks (dilations 1–2–4 and 8–16). Each convolution is followed by layer normalisation [Ba 2016], a ReLU activation and dropout (rate 0.2). Each block adds a 1 × 1 convolution of its input to its output (residual connection) [He 2016]. A dense layer maps the 16 channels at the final time step to the V̇O₂ estimate. The receptive field is 1 + (7 − 1) × (1 + 2 + 4 + 8 + 16) = 187 s, which sets the input window. Depending on the number of inputs, the model has 7.9–8.7 × 10³ trainable parameters. Kernels were initialised with Glorot-uniform initialisation [Glorot 2010] and biases with zeros.

The architecture was fixed in advance and identical for every configuration. Its hyperparameters were not tuned in this study.

## 2.9 Training procedure

One training procedure was fixed before the main analyses and applied identically to every model in every analysis. Only the training data differed (Table 5).

**Table 5.** Training procedure.

| Setting | Value |
|---|---|
| Loss | mean squared error on standardised V̇O₂ |
| Optimiser | Adam [Kingma 2015], β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁷, constant learning rate |
| Learning rate × epochs | **[chosen in the pilot, §2.10]** |
| Batch size | 32 |
| Sample order | reshuffled every epoch |
| Validation set, early stopping | none |
| Model checkpoint | epoch with the lowest mean training loss |
| Random seeds | 5 per configuration and fold; all retained |

Each training example is one target second with its 187-s input window. Every model was trained for the full number of epochs, and the weights of the epoch with the lowest mean training loss (averaged over batches, with dropout active) were retained. No validation data were set aside: every non-test participant contributed to training, and no decision about a model was based on test data.

**Scaling.** All scalers were fitted within each cross-validation fold on the training participants only, and then applied unchanged to the test participant. Work rate was scaled to [0, 1] with min–max scaling; all other continuous inputs and the V̇O₂ target were standardised (z-score); sex was not scaled. Input scalers were fitted on the seconds that the training windows read; the target scaler was fitted on the training target seconds. Predictions were transformed back to mL·min⁻¹ before any metric was computed.

**Random seeds.** Each configuration was trained with five seeds per fold. The seed determined the initial weights and the order of training examples. All five models were retained, and no seed was ever selected over another. Results therefore reflect the expected performance of a single training run, together with its variability across random initialisations.

**Implementation.** Models were implemented in PyTorch. For efficiency, all models of one fold (46 configurations × 5 seeds) were trained simultaneously as independent branches of one network, using grouped convolutions with per-branch normalisation and loss. Each branch had its own parameters, received only its own configuration's inputs and gradients, and was mathematically equivalent to a separately trained model. After training, each branch was extracted into a standalone model, and all predictions were made with the standalone models. Deterministic algorithms were enforced, so a training run is reproducible on the same hardware and software.

## 2.10 Training-settings pilot

The learning rate and number of epochs were chosen once, in a pilot, and then fixed for all analyses. The pilot compared five candidate settings (learning rate × epochs). The reference setting was 5 × 10⁻⁵ × 30, used in one preliminary study. The others were 5 × 10⁻⁵ × 10, 2 × 10⁻⁴ × 10, 5 × 10⁻⁴ × 1 (used in the other preliminary study) and 5 × 10⁻⁴ × 3. Because the learning rate was constant, the 10-epoch and 1-epoch settings correspond to the first epochs of the 30- and 3-epoch runs, so three training runs per split were sufficient. Each shorter setting was evaluated at its own lowest-training-loss epoch within its epoch budget.

The pilot used five configurations spanning the smallest and largest input sets (`WR`, `WR+HR+VENT+TSI+ANTHRO`, `WR+HR+VENT+HBDIFF+ANTHRO`, `HR+VENT+TSI+ANTHRO`, `HR+VENT+HBDIFF+ANTHRO`) with two seeds each. It used the same leave-one-subject-out splits as the main analyses (§2.11), in the pooled dataset (27 splits) and within each cohort (14 and 13 splits), so that a setting adequate for the larger pooled training sets would not undertrain the smaller within-cohort ones. For each split, the held-out participant's RMSE per visit was averaged per participant and then over splits and seeds, giving 15 comparisons (3 datasets × 5 configurations). A pre-specified guide rule favoured the setting with the fewest epochs whose RMSE was within 10 mL·min⁻¹ of the best setting in all 15 comparisons. **[Report the chosen setting, whether it followed the guide rule, its RMSE relative to the reference setting, and the time per training.]**

Because the pilot used leave-one-subject-out splits, the held-out participants in the pilot were the same participants later used as test participants. The training setting was therefore selected with information from the test folds. We consider the resulting optimism small, because a single setting was selected from five candidates and applied identically to all 46 configurations, so it cannot favour particular configurations. Nevertheless, absolute errors may be slightly optimistic. **[Authors: keep this paragraph; the departure must be reported.]**

## 2.11 Cross-validation and analyses

All models were evaluated with leave-one-subject-out cross-validation. In each fold, one participant was held out as the test participant, and the model was trained on all included data of the remaining participants. The test participant's data never contributed to training, scaling or any training decision within that fold. Three analyses used the identical procedure:

- **Pooled** (primary): both cohorts, 27 folds of 26 training participants;
- **Within cohort G**: 14 folds of 13 training participants;
- **Within cohort L**: 13 folds of 12 training participants.

The within-cohort analyses provide reference values for the pooled analysis and for later between-cohort comparisons. Training-set sizes therefore differ by design and are reported with the results. Each analysis comprised 46 configurations × 5 seeds per fold, for 12,420 trained models in total.

## 2.12 Performance metrics

All metrics were computed on the test participant's target seconds in mL·min⁻¹.

**Primary metric.** The root-mean-square error (RMSE) between predicted and measured V̇O₂ was computed per visit, together with the mean absolute error (MAE) and the bias (mean of predicted − measured). Visit metrics were averaged per participant, then over the five seeds. For each configuration, we report the mean ± SD across participants of these seed-averaged values and, separately, the SD across seeds of the configuration mean, which quantifies the variability due to random initialisation. Seed-averaged metrics are the mean of the five models' metrics. They estimate the performance of a single trained model and are not the (lower) error of the averaged prediction.

**Agreement.** For each configuration and seed, we computed the Pearson correlation and the repeated-measures correlation (r_rm) [Bakdash 2017] between measured and predicted V̇O₂ over all test seconds, with participant as the repeated factor. We also computed the repeated-measures Bland–Altman bias and 95 % limits of agreement for the case where the true value varies within participants [Bland 2007]. The variance of the differences was estimated from a one-way analysis of variance with participant as factor, with the between-participant component constrained to be non-negative. These metrics are reported as the mean (SD) across seeds. Correlations are reported without p-values, because the per-second observations are strongly autocorrelated.

**VO₂peak.** For ramp tests, peak V̇O₂ was defined as the highest 30-s rolling mean of V̇O₂ over the evaluated seconds, computed in the same way for measured and predicted V̇O₂. We report the difference (predicted − measured) and its absolute value.

**Breakdowns.** All metrics were reported for all data, per cohort and per protocol.

## 2.13 Statistical analysis

**[Planned; to be finalised before the results are inspected.]** Visit-level, seed-averaged RMSE values will be log-transformed and analysed with a linear mixed model with configuration, protocol and cohort as fixed effects and participant as a random intercept **[justify the random-effects structure; consider random slopes]**. Each configuration will be compared with work rate alone (`WR`), with Holm correction for multiple comparisons. Equivalence with `WR` will be tested with two one-sided tests, with an equivalence margin of ±25 mL·min⁻¹ fixed in advance. **[Rule for selecting configurations for further analyses, e.g. the lowest pooled RMSE, unless a simpler configuration is within the equivalence margin.]** **[Software, e.g. R lme4/emmeans or Python statsmodels; significance level α = 0.05.]**

## 2.14 Sensitivity analysis

**[Optional; decide whether to run and report.]** To test whether the shared mask affected the conclusions, all analyses can be repeated with a per-configuration mask. In that case, each configuration uses every segment valid for the target and its own input groups, so configurations without NIRS or chest-strap inputs can use visits that are excluded from the primary analysis (e.g. `WR` alone uses all 55 cohort G visits).

## 2.15 Software, hardware and reproducibility

Data processing and model training used Python 3.11 with PyTorch 2.14, NumPy 2.4, pandas 3.0 and pingouin 0.7 [Vallat 2018]. Models were trained on a single desktop GPU (Intel Arc B580). Each trained fold stores the code version, all settings, the data-quality table's checksum, the scaler parameters, the per-epoch training losses and the trained weights. An automated test suite verified, among other things, that no test-participant data entered scaling or training, that each model's output at second t depends only on inputs before t, that all configurations in a fold were evaluated on identical seconds, that seeds reproduce exactly, and that the breath-by-breath cleaning reproduces the original implementation. **[Code availability statement: repository URL and commit.]**

**Data availability.** **[Source data are not publicly available due to privacy restrictions; available to verified researchers on request from the corresponding author. Confirm with both ethics approvals.]**

---

### Supplementary Table S1 (draft)

All 46 input configurations, by number of domains: see `docs/METHODOLOGY.md` Appendix A.
