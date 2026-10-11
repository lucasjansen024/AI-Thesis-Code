# Methods (draft v0.2, 9 Oct 2026)

> **Notes for the authors (delete before submission)**
> - Describes the pipeline as implemented at commit `7bd3739`; the source of truth is `docs/METHODOLOGY.md`. When that file changes, update this one.
> - Target journal: JSHS or Sports Medicine (`docs/paper/README.md`). This is the **long version**. Sections tagged **[main]** stay in the main text and will be shortened to ~1,500–2,000 words for JSHS's 5,000-word limit. Sections tagged **[supplement]** move to Supplementary Methods. Main-text display items are Table 1 and Figure 1 only; the other tables are supplementary (S1–S5), leaving room for results within JSHS's 6-item limit.
> - **[Bold square brackets]** = information not found in the repository, or a decision still pending (`METHODOLOGY.md` §15).
> - Citations: `[Author Year]`, listed in `docs/paper/references.md`; convert to the journal's numbered style later.
> - Terms: *visit* (one protocol performed by one participant); *repetition* (one of the two identical PRBS sequences in a visit); *segment* (the unit of quality assessment: a repetition, or the whole exercise phase); *domain* (input group); *configuration* (set of domains).

---

## 2.1 Participants [main]

Data from two independent studies were pooled (Table 1). Both were performed with the same equipment in the same laboratory **[confirm: Human Performance Laboratory, Vrije Universiteit Amsterdam, for both cohorts]**. **Cohort G** (2024) comprised 14 healthy, recreationally active adults (8 men, 6 women); a fifteenth participant withdrew and is not included. **Cohort L** (2026) comprised 13 healthy adults (8 men, 5 women) without chronic conditions likely to alter the normal response to exercise. Participants were identified by cohort and original study number (G01, G03–G15; L01–L10, L12–L14).

All participants gave written informed consent and were informed of their right to withdraw without consequence. Cohort G was approved by the Natural Sciences & Engineering Sciences (NES) ethics committee **[institution]** (request #240302). Cohort L was approved by the Scientific and Ethical Review Board (VCWE) of the Faculty of Behavioural and Movement Sciences, Vrije Universiteit Amsterdam (VCWE-2026-113). **[Declaration of Helsinki statement, if both approvals allow it.]**

**Table 1.** Participant characteristics.

| | Cohort G (n = 14) | Cohort L (n = 13) | Pooled (n = 27) |
|---|---|---|---|
| Sex (men/women) | 8/6 | 8/5 | 16/11 |
| Age (y) | 25.7 ± 3.4 | 25.4 ± 2.5 | 25.6 ± 2.9 |
| Height (cm) | 180.2 ± 9.3 | 178.0 ± 9.9 | 179.2 ± 9.5 |
| Body mass (kg) | 74.0 ± 9.9 | 69.9 ± 13.6 | 72.0 ± 11.8 |
| Skeletal muscle mass (kg) | 30.7 ± 5.1 | 29.9 ± 6.1 | 30.3 ± 5.5 |
| Peak V̇O₂ (L·min⁻¹) | 3.47 ± 0.64 | 3.11 ± 0.87 | 3.30 ± 0.77 |
| Peak V̇O₂ (mL·kg⁻¹·min⁻¹) | 47.0 ± 7.4 | 44.1 ± 7.3 | 45.6 ± 7.4 |
| Peak power (W) | 348 ± 64 | 298 ± 87 | 324 ± 79 |
| GET (L·min⁻¹) | 2.22 ± 0.38 | 2.10 ± 0.47 | 2.16 ± 0.42 |
| GET (W) | 153 ± 31 | 142 ± 41 | 148 ± 36 |
| GET (% peak V̇O₂) | 64 ± 4 | 69 ± 7 | 66 ± 6 |

Values are mean ± SD. Age and height come from the metabolic-cart profile. Body mass and skeletal muscle mass come from the bioelectrical-impedance scale before the ramp test. Peak values and the gas exchange threshold (GET) come from the ramp test (§2.3). V̇O₂, oxygen uptake.

## 2.2 Experimental design [main]

Each participant completed four visits on separate occasions, all on the same electronically braked cycle ergometer (LC7TT, Monark Exercise AB, **[city — Lucas's thesis says Varberg; confirm]**, Sweden). The first visit was always a ramp-incremental test. Its results were used to prescribe the intensities of the three subsequent visits, which were performed in random order. In cohort G, these were three pseudorandom binary sequence (PRBS) visits with different intensity combinations. In cohort L, they were one PRBS visit, one interval visit and one step visit (§2.3).

In cohort G, all four visits were completed within 1–23 days, and ten pairs of visits took place on the same day, 2.2–7.6 h apart. In cohort L, visits were spread over 9–39 days and were at least 1 day apart. Tests took place between 09:00 and 19:30, at 19.3 ± 0.9 °C (range 18–22 °C) and 54 ± 7 % relative humidity, as recorded by the metabolic cart. Participants in cohort L were asked to avoid strenuous exercise and alcohol for 24 h, caffeine on the day of testing and food for 2 h before each visit **[confirm whether the same instructions applied to cohort G]**. Cadence was self-selected between 70 and 90 rpm and kept the same across a participant's visits.

Every visit began with 2 min of quiet seated rest on the ergometer with all sensors attached, followed by a warm-up and the exercise phase. The rest phase served only to determine resting heart rate (§2.5.6). The end of the warm-up served as input history for the models but was never predicted. All prediction targets came from the exercise phase.

## 2.3 Exercise protocols [main; Table S1 supplement]

Work-rate profiles are shown in Figure 1A and timelines in Table S1. **[Figure 1: (A) work-rate profiles of the seven protocols for one participant per cohort; (B) schematic of the analysis pipeline: sensors → preprocessing → quality control and shared mask → 187-s input window → TCN → leave-one-subject-out evaluation.]**

**Ramp-incremental test (both cohorts).** After 2 min of rest, participants cycled for 4 min at 30 W. Work rate then increased by 20, 25, 30 or 35 W·min⁻¹ until task failure. The ramp rate was chosen per participant from body size and self-reported activity level, aiming for 8–12 min of ramp exercise. Actual ramp durations were 11.5 ± 1.3 min in cohort G and 10.4 ± 1.9 min in cohort L. Task failure was defined as a cadence below 60 rpm despite strong verbal encouragement **[cohort L: for more than 5 s; confirm cohort G]**.

**Determination of intensities (both cohorts).** Both cohorts used the same procedure and spreadsheet, on 10-s averaged ramp data.
- *GET.* Identified visually as the V̇O₂ at which V̇CO₂ increased disproportionately relative to V̇O₂ (V-slope). This was confirmed by an increase in the ventilatory equivalent and end-tidal partial pressure for O₂ without an increase in those for CO₂ [Goulding 2017; Wasserman 1984].
- *Peak V̇O₂.* The highest 30-s average.
- *Conversion to power.* V̇O₂ was regressed linearly on time over the ramp. The mean response time of V̇O₂ (MRT; 36 ± 19 s) was the delay at which this regression line reached the baseline V̇O₂ of the last 60 s of 30-W cycling. A V̇O₂ value was then converted to the ramp power at its regression time minus the MRT.
- *Intensity levels.* Intensities between GET and peak are expressed as Δx % = GET + x % × (peak V̇O₂ − GET), converted to power in the same way. Fractions of GET (90 %, 80 %, 60 %) refer to the GET power.

**PRBS visits (cohort G).** Work rate switched between two levels according to a PRBS with 30-s units, giving runs of 30–90 s at one level. After rest, the warm-up was the first 3.5 min of the sequence at the visit's levels. Two repetitions of the 16-min sequence followed (32 min). The three visits paired different levels, chosen to elicit responses within the moderate domain, spanning the moderate and heavy domains, and from the domain boundary into the heavy domain:
- *Easy* (low→moderate): 25 W ↔ 90 % GET;
- *Medium* (low→heavy): 25 W ↔ Δ30 %;
- *Hard* (moderate→heavy): GET ↔ Δ30 %.

**PRBS visit (cohort L).** After rest and a 3.5-min warm-up at 80 % GET, work rate switched between 25 W and Δ40 %. The PRBS had 31 units of 30 s, generated by a linear-feedback shift register (runs of 30–150 s). The 15.5-min sequence was performed twice (31 min).

**Interval visit (cohort L).** After rest and the same warm-up, participants performed 10 cycles of 2 min at 25 W and 1 min at Δ70 %, followed by 1 min at 25 W (31 min).

**Step visit (cohort L).** After rest and the same warm-up, work rate followed 6 min at 80 % GET, 6 min at Δ30 %, 6 min at 80 % GET, 3 min at Δ60 %, 5 min at 25 W and 5 min at 60 % GET (31 min).

All submaximal work rates were executed by the ergometer from stored protocol files, and the levels above were checked against those files for this analysis.

Two cohort G visits were repeated after a partial failure, and only the complete repetition was used. One ramp test was aborted and repeated, and only the repeat was used. One participant of cohort G did not perform the Medium visit. This gives 55 candidate visits in cohort G and 52 in cohort L.

**Table S1.** Protocol timelines as analysed (times from the start of the recording).

| Cohort | Protocol | Rest | Warm-up | Exercise phase (prediction targets) | Quality segments |
|---|---|---|---|---|---|
| G, L | Ramp | 0–120 s | 120–360 s, 30 W | 360 s → task failure | 1 |
| G | Easy, Medium, Hard (PRBS) | 0–120 s | 120–330 s, PRBS | 330–2250 s (2 × 960 s) | 2 (repetitions) |
| L | PRBS | 0–120 s | 120–330 s, 80 % GET | 330–2190 s (2 × 930 s) | 2 (repetitions) |
| L | Interval, Step | 0–120 s | 120–330 s, 80 % GET | 330–2190 s | 1 |

## 2.4 Measurements [main, shortened; details supplement]

**Pulmonary gas exchange (ground truth).** Gas exchange was measured breath-by-breath with a metabolic cart (Quark CPET, COSMED, Rome, Italy). Before each test, the gas analysers were calibrated with gases of known concentration and the turbine with a 3-L syringe (Hans Rudolph, Kansas City, MO, USA), following the manufacturer's instructions. The cart also recorded heart rate (HR) from its own chest strap **[model]**, which served only as a backup HR source (§2.5.6). The cart's minute ventilation and breathing frequency were never used as model inputs, because they are derived from the same flow signal as V̇O₂.

**Chest strap.** HR, minute ventilation (V̇E) and breathing rate (BR) were recorded with a Tymewear VitalPro strap (Tymewear, Boston, MA, USA). It combines an HR sensor with a respiratory sensor that estimates ventilation from chest-wall movement. The strap was placed around the thorax directly below the pectoral muscles (men) or below the breasts (women). It was calibrated in the companion app with three full inhalation–exhalation cycles, but not against the metabolic cart. HR was exported at 1 Hz, and V̇E and BR per breath.

**Near-infrared spectroscopy (NIRS).** Oxygenation of the right vastus lateralis was measured with a continuous-wave, spatially resolved NIRS device (PortaMon; Artinis Medical Systems, **[Elst/Zetten — confirm]**, the Netherlands; cohort L used the MKIII version).
- *Device.* Three light sources at 30, 35 and 40 mm from the detector, at wavelengths of ~760 and ~850 nm.
- *Signals.* Relative changes in oxygenated (O₂Hb), deoxygenated (HHb) and total haemoglobin per source. Also the tissue saturation index (TSI, %), with a fit factor that indicates how well the three distances agree.
- *Placement.* On the muscle belly, midway between the greater trochanter and the lateral femoral epicondyle **[confirm for cohort G: "midway along the belly"]**. Fixed with tape and bandages (cohort L: also a blackout sleeve) to limit movement and ambient light.
- *Recording.* All recordings were exported at 10 Hz. An arterial occlusion was performed after exercise in both cohorts for NIRS calibration **[confirm protocol]**. These data fall outside the analysed time window and were not used.

**Body composition.** Before each visit, body mass and skeletal-muscle percentage were measured with a bioelectrical-impedance (BIA) smart scale (Sacoma, Amsterdam, the Netherlands; Fitdays app). Age and sex were taken from the participant profile in the metabolic-cart software.

## 2.5 Data processing [main: short summary; supplement: §2.5.1–2.5.8 in full]

Both cohorts were processed from the raw exports with one pipeline (Python; §2.15). No processed data or results from the original studies were reused.

### 2.5.1 File selection and synchronisation

Files were selected by explicit rules, recorded per visit. Only breath-by-breath gas-exchange exports were used; time-averaged exports were never used. The metabolic cart's clock served as the reference time base. Chest-strap data were already on this clock in cohort L. In cohort G, they were shifted by a per-visit offset, determined in the original study by aligning the strap's and the cart's ventilation. NIRS recordings were aligned using the event marker set at a known protocol time. When a file contained several markers (8 restarted recordings), the last one was used. No manual per-visit corrections were applied.

### 2.5.2 Time base and phases

Each visit was represented on a 1-Hz grid from the start of rest to the end of exercise. Phases were assigned from the nominal protocol timeline (Table S1), not from manual phase marks. The end of a ramp test was the last breath with a cadence of at least 60 rpm.

### 2.5.3 Ground truth: V̇O₂

Breath-by-breath V̇O₂ was cleaned in three steps, identical for both cohorts:
1. breaths that differed by 3 SD or more from the mean of the preceding five breaths were removed;
2. a centred 5-breath median filter was applied;
3. values were linearly interpolated to the 1-Hz grid.

Interpolation was not trusted across long gaps: a second was marked as missing when the raw breaths on either side of it were more than 20 s apart.

### 2.5.4 Work rate

The work-rate input was the work rate prescribed by the protocol, reconstructed second by second from the ergometer's protocol file (submaximal visits) or from the ramp settings (ramps). The power logged by the metabolic cart was not used: it is recorded per breath, so it is irregularly sampled and misaligned with the load changes. The reconstruction agreed closely with the logged power (median absolute difference 0–1 W over the exercise phase of every ramp).

Three visits needed substitutes. The two repeated cohort G visits lacked their own protocol file, so the file of the original attempt (same protocol) was used. One cohort L PRBS visit had no protocol file. Its timing was taken from the identical PRBS file of another participant, and its high and warm-up levels from the median logged power during those phases.

### 2.5.5 Ventilation

V̇E and BR were taken from the strap's breath-by-breath output and cleaned with the same three steps as V̇O₂. The one difference is that the median filter was trailing (the current and four preceding breaths), so that inputs depend only on past breaths. A second was marked as missing when raw breaths were more than 20 s apart. Ventilation counted as available only when both V̇E and BR were available.

### 2.5.6 Heart rate and heart-rate reserve

HR was taken from the chest strap when it passed the quality check in §2.6. Otherwise, the cart's chest-strap HR was used if it passed the same check (5 visits). HR was linearly interpolated to 1 Hz without smoothing. Heart-rate reserve (HRR) was computed per second as

HRR(t) = (HR(t) − HR_rest) / (HR_peak − HR_rest),

where:
- HR_rest is the mean HR over the first 120 s (rest) of the same visit, from the same source;
- HR_peak = 208 − 0.7 × age [Tanaka 2001], for all participants.

This definition uses no information from other visits or from the participant's maximal test, and it is available in practice without a laboratory test. For one visit, the wrong participant profile had been selected in the cart software, so age was taken from the same participant's previous visit.

### 2.5.7 Muscle oxygenation

Two NIRS variables were derived:
- TSI;
- the haemoglobin difference (Hb-diff = O₂Hb − HHb) from the source **[intended to be the 40-mm source; channel-to-distance mapping to be verified]**.

Both were used as exported: they were not normalised to the occlusion or to the visit's own range, and no baseline was subtracted. The only scaling was the fold-wise standardisation in §2.9. The zero point of the relative haemoglobin signals is arbitrary per recording, but subtracting the resting mean made little visible difference compared with the exercise-induced changes. All-zero rows and a non-physiological first row in some exports were removed. The 10-Hz signals were averaged to 1 Hz (mean of the samples within each second). A second without NIRS samples was marked as missing. The fit factor was used only for quality control (§2.6).

### 2.5.8 Anthropometrics

Sex was coded 1 for male and 0 for female. Skeletal muscle mass (SMM, kg) was computed as the scale's skeletal-muscle percentage multiplied by body mass; this is distinct from the scale's separate "muscle mass" output. SMM was taken from the measurement on the day of the visit and was constant within a visit. For one visit without a same-day measurement, the nearest earlier measurement (6 days before) was used. Sex and SMM were repeated at every second of the visit.

## 2.6 Data quality and inclusion [main: principle and result; supplement: criteria in full, Table S2]

Data quality was assessed **before any model was trained**, with explicit rules per sensor and per segment of the exercise phase. The verdicts were stored in one table, which the training code reads but never modifies. The aim was to remove recordings that were missing or physiologically implausible, not data that are hard to predict, so no rule refers to prediction error.

**Segments and sensor groups.** Validity was decided per *segment*: each PRBS repetition separately, and otherwise the whole exercise phase of the visit (161 segments). Within a segment, it was decided for each *sensor group*: V̇O₂ (target), work rate, HR, ventilation, NIRS and static (sex and SMM). A group was valid or invalid for a whole segment, never for single seconds, because a corrupted second would also enter the input history of the following 187 s. Only exercise seconds were assessed: the warm-up serves only as input history, so its quality never invalidated a segment.

**Automatic criteria.**
- *Presence and gaps (all groups).* A group was invalid if its recording was missing, or if the longest gap without samples in the segment exceeded 20 s.
- *Work rate.* Valid if the protocol definition was available. Three visits used the substitutes in §2.5.4.
- *Heart rate.* Each source (strap first, then the cart's chest strap) had to meet three criteria:
  - an HR range (95th − 5th percentile) of at least 20 bpm;
  - a Pearson correlation of at least 0.5 with 30-s smoothed V̇O₂;
  - data for at least 80 % of the rest phase, which is needed for HR_rest.

  The range and correlation were computed from the start of rest to the end of exercise. These criteria detect sensors that record a constant or unrelated signal, which a coverage check alone misses. They start at rest because genuine HR varies too little during heavy PRBS exercise for an exercise-only check. They reproduced the original study's manual HR-source annotations exactly.
- *NIRS.* In addition to the gap rule, at least 95 % of the segment's NIRS samples needed a fit factor of at least 99 %. No validated threshold exists for this index. A stricter per-sample threshold (99.9 %) would have excluded nine cohort L visits whose signals were otherwise good. On these data, this criterion excluded no segment.

**Visual NIRS review.** One investigator also reviewed the NIRS signals visually, before any model had been trained. A review could exclude a segment that passed the automatic checks, but could not include a segment that failed them. Four segments were excluded on review:
- both repetitions of one visit;
- the second repetition of another visit;
- the first repetition of a third visit, whose warm-up contained a TSI artefact that would have entered the input history of that repetition's first targets.

**Resulting dataset.** All 52 cohort L visits and 47 of 55 cohort G visits were included (Table S2; **[Figure S1: data-flow diagram]**). The eight exclusions were due to:
- a missing chest-strap recording (4 ramp tests);
- unusable HR from both sources (2 visits) or from the only available source (1 visit);
- unusable NIRS (1 visit).

Two further cohort G visits contributed one repetition each. In one, the gas-exchange recording stopped during the second repetition; in the other, the first repetition was excluded on NIRS review. HR came from the cart's chest strap in 5 visits.

**Table S2.** Included visits and prediction targets per protocol.

| Cohort | Protocol | Visits included | Target seconds |
|---|---|---|---|
| G | Ramp | 9/14 | 5,925 |
| G | Easy | 12/14 | 23,030 |
| G | Medium | 13/13 ¹ | 23,979 |
| G | Hard | 13/14 ² | 23,999 |
| L | Ramp | 13/13 | 8,061 |
| L | PRBS | 13/13 | 24,095 |
| L | Interval | 13/13 | 24,146 |
| L | Step | 13/13 | 24,147 |
| | **Total** | **99/107** | **157,382 (43.7 h)** |

¹ One visit with repetition 2 only. ² One visit with repetition 1 only.

**Prediction targets and shared mask.** A second t was a prediction target only if all of the following held:
1. it lay in the exercise phase;
2. its segment was valid for **all six** sensor groups;
3. V̇O₂ was available at t;
4. every input group had data (no gap > 20 s) throughout the preceding 187 s, which is the model's input window.

The input window could extend into the warm-up, or into a neighbouring repetition that was invalid for another reason, as long as the data were present. Because every configuration required all groups (the *shared mask*), all input configurations were trained and evaluated on exactly the same seconds. Differences between configurations therefore cannot arise from differences in the data. This gave 157,382 target seconds (cohort G 76,933; cohort L 80,449). Of the exercise seconds in valid segments, condition 4 removed only 205 s, all at the end of recordings.

## 2.7 Input domains and configurations [main; Tables S3 and S5 supplement]

Inputs were grouped into five domains, each corresponding to one sensor or physiological system (Table S3). Within a domain, features were always used together (e.g. HR always with HRR). The NIRS domain entered through exactly one of its two variables, TSI or Hb-diff, because both derive from the same measurement. The anthropometric domain has no time course, so it was only used together with at least one time-varying domain.

**Table S3.** Input domains.

| Domain | Features | Sensor | Scaling |
|---|---|---|---|
| WR | work rate | ergometer (protocol) | min–max |
| HR | HR, HRR | chest strap (cart chest strap as backup) | z-score |
| VENT | V̇E, BR | chest strap (respiratory sensor) | z-score |
| NIRS | TSI *or* Hb-diff | NIRS, vastus lateralis | z-score |
| ANTHRO | sex, SMM | cart profile, BIA scale | sex: none; SMM: z-score |

All combinations were evaluated. WR, HR, VENT and at most one NIRS variable give 23 non-empty combinations; each was used with and without ANTHRO, giving **46 configurations** (Table S5) with 1–8 input features. Configurations are named by their domains, with NIRS written as its variable (e.g. `WR+HR+TSI+ANTHRO`).

## 2.8 Model [main]

We used a causal temporal convolutional network (TCN) [Bai 2018]. Its architecture was introduced for V̇O₂ prediction by Amelard et al. [Amelard 2021] and used in both preliminary studies (Figure 1B). The model maps the inputs of the preceding 187 s (seconds t − 187 to t − 1) to V̇O₂ at second t. Because the input at t itself is not used, the model is causal and could run in real time.

The network has five causal dilated one-dimensional convolutions (kernel size 7, 16 filters, dilation factors 1, 2, 4, 8 and 16), arranged in two residual blocks (dilations 1–2–4 and 8–16).
- Each convolution is followed by layer normalisation [Ba 2016], a ReLU activation and dropout (rate 0.2).
- Each block adds a 1 × 1 convolution of its input to its output (residual connection) [He 2016].
- A dense layer maps the 16 channels at the final time step to the V̇O₂ estimate.

The receptive field is 1 + (7 − 1) × (1 + 2 + 4 + 8 + 16) = 187 s, which sets the input window. The model has 7.9–8.7 × 10³ trainable parameters, depending on the number of inputs. Kernels were initialised with the Glorot-uniform scheme [Glorot 2010] and biases with zeros. The architecture was fixed in advance, identical for every configuration, and not tuned in this study.

## 2.9 Training procedure [main: principle; Table S4 supplement]

One training procedure was fixed before the main analyses and applied identically to every model; only the training data differed (Table S4).

**Table S4.** Training procedure.

| Setting | Value |
|---|---|
| Loss | mean squared error on standardised V̇O₂ |
| Optimiser | Adam [Kingma 2015], β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁷, constant learning rate |
| Learning rate × epochs | 5 × 10⁻⁵ × 30 (sensitivity analysis: 5 × 10⁻⁴ × 1; §2.10, §2.14) |
| Batch size | 32 |
| Sample order | reshuffled every epoch |
| Validation set, early stopping | none |
| Model checkpoint | epoch with the lowest mean training loss |
| Random seeds | 5 per configuration and fold; all retained |

Each training example is one target second with its 187-s input window. Every model was trained for the full number of epochs. The weights of the epoch with the lowest mean training loss (averaged over batches, with dropout active) were retained. No validation data were set aside: every non-test participant contributed to training, and no decision about a model used test data.

**Scaling.** All scalers were fitted within each cross-validation fold on the training participants only, then applied unchanged to the test participant.
- Work rate was scaled to [0, 1] (min–max).
- All other continuous inputs and the V̇O₂ target were standardised (z-score).
- Sex was not scaled.

Input scalers were fitted on the seconds read by the training windows, and the target scaler on the training target seconds. Predictions were transformed back to mL·min⁻¹ before any metric was computed.

**Random seeds.** Each configuration was trained with five seeds per fold. The seed determined the initial weights and the order of training examples. All five models were retained, and no seed was selected over another. Results therefore reflect the expected performance of a single training run and its variability across random initialisations.

**Implementation [supplement].** Models were implemented in PyTorch. For efficiency, all models of one fold (46 configurations × 5 seeds) were trained simultaneously as independent branches of one network, using grouped convolutions with per-branch normalisation and loss. Each branch had its own parameters and received only its own configuration's inputs and gradients, so it was mathematically equivalent to a separately trained model. After training, each branch was extracted into a standalone model, and all predictions were made with the standalone models. Deterministic algorithms were enforced, so a training run is exactly reproducible on the same hardware and software.

## 2.10 Training settings [main]

The learning rate (5 × 10⁻⁵) and number of epochs (30) were those of one preliminary study and the default of the analysis plan, fixed before any result of the main analyses was available. A planned pilot comparing cheaper settings was stopped early, before any setting was chosen; the default was therefore retained. As a sensitivity analysis (§2.14), all analyses were repeated with the setting of the other preliminary study (5 × 10⁻⁴ × 1 epoch).

## 2.11 Cross-validation and analyses [main]

All models were evaluated with leave-one-subject-out cross-validation. In each fold, one participant was held out for testing, and the model was trained on all included data of the remaining participants. The test participant's data never contributed to training, scaling or any training decision within that fold. Three analyses used the identical procedure:

- **Pooled** (primary): both cohorts, 27 folds of 26 training participants;
- **Within cohort G**: 14 folds of 13 training participants;
- **Within cohort L**: 13 folds of 12 training participants.

The within-cohort analyses provide reference values for the pooled analysis and for later between-cohort comparisons. Training-set sizes therefore differ by design; they are reported with the results. Each analysis comprised 46 configurations × 5 seeds per fold, for 12,420 trained models in total.

## 2.12 Performance metrics [main]

All metrics were computed in mL·min⁻¹ on the test participant's target seconds.

**Primary metric.** The root-mean-square error (RMSE) between predicted and measured V̇O₂ was computed per visit, together with the mean absolute error (MAE) and the bias (mean of predicted − measured). Visit metrics were averaged per participant, then over the five seeds. For each configuration, we report:
- the mean ± SD across participants of these seed-averaged values;
- the SD across seeds of the configuration mean, which quantifies the variability due to random initialisation.

Seed-averaged metrics are the mean of the five models' metrics. They estimate the performance of a single trained model, not the (lower) error of the averaged prediction.

**Agreement.** For each configuration and seed, we computed the Pearson correlation and the repeated-measures correlation (r_rm) [Bakdash 2017] between measured and predicted V̇O₂ over all test seconds, with participant as the repeated factor. We also computed the repeated-measures Bland–Altman bias and 95 % limits of agreement for the case where the true value varies within participants [Bland 2007]. The variance of the differences was estimated from a one-way analysis of variance with participant as factor, with the between-participant component constrained to be non-negative. These metrics are reported as mean (SD) across seeds. Correlations are reported without p-values, because the per-second observations are strongly autocorrelated.

**Peak V̇O₂.** For ramp tests, peak V̇O₂ was the highest 30-s rolling mean over the evaluated seconds, computed the same way for measured and predicted V̇O₂. We report the difference (predicted − measured) and its absolute value.

**Breakdowns.** All metrics are reported for all data, per cohort and per protocol.

## 2.13 Statistical analysis [main]

**[Planned; to be finalised before the results are inspected.]** Visit-level, seed-averaged RMSE will be log-transformed and analysed with a linear mixed model with configuration, protocol and cohort as fixed effects and participant as a random intercept **[justify the random-effects structure; consider random slopes]**. Each configuration will be compared with work rate alone (`WR`), with Holm correction for multiple comparisons. Equivalence with `WR` will be tested with two one-sided tests against a margin of ±25 mL·min⁻¹, fixed in advance. **[Rule for selecting configurations for further analyses, e.g. the lowest pooled RMSE unless a simpler configuration is within the equivalence margin.]** **[Software, e.g. R lme4/emmeans or Python statsmodels; α = 0.05.]**

## 2.14 Sensitivity analyses [supplement]

**Training setting.** All three analyses (pooled and within each cohort) were repeated with a learning rate of 5 × 10⁻⁴ for one epoch (§2.10), with everything else unchanged: the same folds, configurations, seeds, scaling and evaluation. This shows whether the conclusions depend on the amount of training. The pilot (five settings, five configurations, leave-one-subject-out) was stopped after 8 of 54 planned splits because the settings differed by less than the variation between seeds; the sensitivity setting was chosen after these partial results, whose held-out participants were later test participants.

**Shared mask. [Optional; decide whether to run and report.]** To test whether the shared mask affected the conclusions, all analyses can be repeated with a per-configuration mask. Each configuration then uses every segment valid for the target and its own input groups. Configurations without NIRS or chest-strap inputs can then use visits excluded from the primary analysis; for example, `WR` alone uses all 55 cohort G visits.

## 2.15 Software, hardware and reproducibility [main: 2 sentences; rest supplement]

Data processing and model training used Python 3.11 with PyTorch 2.14, NumPy 2.4, pandas 3.0 and pingouin 0.7 [Vallat 2018], on a single desktop GPU (Intel Arc B580). Each trained fold stores the code version, all settings, the data-quality table's checksum, the scaler parameters, per-epoch training losses and the trained weights. An automated test suite verified that:
- no test-participant data entered scaling or training;
- each model's output at second t depends only on inputs before t;
- all configurations in a fold were evaluated on identical seconds;
- seeds reproduce exactly;
- the breath-by-breath cleaning reproduces the original implementation.

**[Code availability: repository URL and commit.]**

**Data availability.** **[Source data are not publicly available due to privacy restrictions; they are available to verified researchers on request from the corresponding author. Confirm against both ethics approvals.]**

---

### Table S5 (draft)

All 46 input configurations by number of domains: see `docs/METHODOLOGY.md`, Appendix A.
