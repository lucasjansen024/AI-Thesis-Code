# Results (draft v0.1, 14 Oct 2026)

> **Notes for the authors (delete before submission)**
> - First results draft, written by Claude from the pipeline outputs on 14 Oct 2026 and not yet reviewed by the dev. Every number was copied or recomputed (awk) from a file under `outputs/`; the file behind each block is listed under *Sources* at the end. A number that could not be sourced is marked **[CHECK: …]**.
> - **Descriptive only** (`METHODOLOGY.md` §12; methods §2.15): mean ± SD across participants, and for Aim 2 the mean difference with a 95 % t-CI. No hypothesis tests were run, so there are no p-values and nothing is called "significant"; a CI that excludes 0 is stated as such and nothing more. Interpretation is left for the discussion: each subsection has at most one framing sentence.
> - **Aim 2 (§3.5) is interim**: 2b Step is still training. See the note at the top of §3.5.
> - Conventions as in `02_methods.md`: participant = unit of analysis; values in mL·min⁻¹; RMSE = seed-averaged RMSE (§2.13); configuration names in backticks. In Aim 1, "Δ vs `WR`" is the difference between the pooled means, without a CI (the outputs give none).
> - Display items: Table 2, Figure 2 and Figure 3 (main text; with Table 1 and Figure 1 from the methods, 5 of JSHS's 6). Display item 6 (error by phase) is undecided and is **not** drafted; the residual results are in §3.4 and Table S10. Supplementary tables continue the methods' numbering (S8–S11); the order is provisional.
> - Main-text word count (excluding tables, legends and these notes): §3.1–3.4 ≈ 1,260 (of which §3.4, residuals, ≈ 300); §3.5 ≈ 330, plus the interim note.

---

## 3.1 Input configurations (Aim 1) [main]

To identify which inputs are needed, all 46 configurations were compared on the same 157,382 target seconds from 99 visits of 27 participants (Figure 2; Table 2; all configurations in Table S8).

**Figure 2.** RMSE of predicted V̇O₂ per input configuration (Aim 1). Large dots: pooled leave-one-subject-out analysis, mean ± SD across participants (n = 27) of the seed-averaged RMSE. Small markers: mean RMSE in the within-cohort analyses (diamonds, cohort G only, n = 14; squares, cohort L only, n = 13). Dashed line: `WR` in the pooled analysis (186 mL·min⁻¹). Rows are grouped as single inputs, single inputs with anthropometrics, work rate plus one physiological domain, and the configuration without work rate with the lowest RMSE. NIRS is shown as TSI; the Hb-diff configurations are in Table S8.

**Work rate alone.** With work rate as the only input (`WR`), the RMSE was 186 ± 30 mL·min⁻¹ (mean ± SD across participants; MAE 154 ± 29 mL·min⁻¹). No other configuration had a lower mean RMSE. The other 23 configurations containing work rate ranged from 196 to 220 mL·min⁻¹ (+10 to +34 vs `WR`), and the 22 configurations without work rate from 362 to 483 mL·min⁻¹ (Table S8). The SD across seeds of the configuration mean was 1.0 mL·min⁻¹ for `WR` and 0.8–16.2 mL·min⁻¹ across all 46 configurations, smaller than the SD across participants in every configuration.

**Single physiological inputs.** Each physiological domain on its own gave an RMSE 2.3 to 2.6 times that of `WR`: `VENT` 431 ± 142, `TSI` 436 ± 164, `HBDIFF` 452 ± 165 and `HR` 481 ± 142 mL·min⁻¹ (Figure 2; Table S8). Their repeated-measures correlations were 0.762–0.785, against 0.957 for `WR`.

**Anthropometrics.** Adding `ANTHRO` changed the RMSE in opposite directions with and without work rate. With work rate, it raised the mean RMSE in all 12 pairs, by 7–22 mL·min⁻¹ (`WR+ANTHRO` 200 ± 52 mL·min⁻¹, +13 vs `WR`). Without work rate, it lowered the mean RMSE in 7 of 11 pairs. It lowered it in five of the six pairs containing HR, by 11–78 mL·min⁻¹ (e.g. `HR` 481 → `HR+ANTHRO` 406; `HR+VENT` 483 → `HR+VENT+ANTHRO` 405), and in both NIRS-only pairs (by 18 and 19). It raised it for `HR+VENT+HBDIFF` (+14) and for the three ventilation configurations without HR (+18 to +26).

**Work rate plus physiology.** Adding one physiological domain to `WR` raised the mean RMSE by 10–18 mL·min⁻¹: `WR+VENT` 196 ± 39 (+10), `WR+TSI` 198 ± 41 (+12), `WR+HBDIFF` 202 ± 40 (+16) and `WR+HR` 204 ± 41 mL·min⁻¹ (+18). The seven configurations combining work rate with two or three physiological domains, without `ANTHRO`, lay between 197 and 206 mL·min⁻¹ (+11 to +19). Per participant, `WR+VENT` had a lower RMSE than `WR` in 10 of 27 participants, `WR+HR` in 11 and `WR+TSI` in 9. `WR+VENT` was therefore the work-rate-plus-physiology configuration carried forward to Aim 2 (§2.12).

**Best configuration without work rate.** Among the 22 configurations without work rate, `HR+VENT+TSI+ANTHRO` had the lowest RMSE, 362 ± 158 mL·min⁻¹ (+176 vs `WR`). It was followed by `HR+TSI+ANTHRO` (372 ± 174), `HR+VENT+HBDIFF` (376 ± 112) and `VENT+TSI` (378 ± 157 mL·min⁻¹). The SD across participants of `HR+VENT+TSI+ANTHRO` was about five times that of `WR` (158 vs 30 mL·min⁻¹), and its RMSE was lower than that of `WR` in 4 of 27 participants.

**Table 2.** Agreement of the key configurations with measured V̇O₂ (Aim 1).

| Configuration | RMSE | ΔRMSE vs WR | Bias | r_rm | 95 % LoA | VO₂peak error | RMSE, Cohort G only (n = 14) | RMSE, Cohort L only (n = 13) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `WR` | 186 ± 30 | — | −1 ± 110 | 0.957 | −389 to 366 | +216 ± 263 | 194 ± 36 | 196 ± 29 |
| `WR+VENT` | 196 ± 39 | +10 | +2 ± 119 | 0.953 | −401 to 390 | +169 ± 280 | 206 ± 81 | 200 ± 34 |
| `HR+VENT+TSI+ANTHRO` | 362 ± 158 | +176 | −62 ± 303 | 0.881 | −846 to 729 | −110 ± 517 | 323 ± 141 | 369 ± 158 |
| `HR` | 481 ± 142 | +295 | −8 ± 371 | 0.762 | −1021 to 993 | −128 ± 868 | 425 ± 119 | 626 ± 238 |
| `HR+ANTHRO` | 406 ± 176 | +220 | −15 ± 360 | 0.858 | −917 to 891 | −185 ± 539 | 285 ± 103 | 430 ± 184 |

Values in mL·min⁻¹ (r_rm unitless). Pooled leave-one-subject-out (n = 27) unless marked. RMSE, bias and VO₂peak error: mean ± SD across participants of the seed-averaged values. ΔRMSE: difference of the mean RMSE to WR. r_rm and the repeated-measures Bland–Altman 95 % limits of agreement: over all target seconds, mean of 5 seeds. VO₂peak error: predicted − measured highest 30-s mean, ramp visits only. Within-cohort columns: leave-one-subject-out within that cohort.

## 3.2 Cohort comparison (Aim 1) [main]

The within-cohort analyses show whether the pattern holds in each cohort, and how the pooled model compares with models trained on one cohort only (Figure 2, small markers; Table S8).

**Within each cohort.** `WR` had the lowest RMSE in both within-cohort analyses: 194 ± 36 mL·min⁻¹ in cohort G (n = 14) and 196 ± 29 mL·min⁻¹ in cohort L (n = 13). Configurations without work rate differed more between the cohorts, for example `HR` 425 ± 119 vs 626 ± 238 mL·min⁻¹ and `HR+VENT` 336 ± 121 vs 653 ± 183 mL·min⁻¹ (cohort G vs cohort L). The configuration without work rate with the lowest RMSE was `HR+ANTHRO` in cohort G (285 ± 103 mL·min⁻¹) and `HR+VENT+HBDIFF+ANTHRO` in cohort L (325 ± 131 mL·min⁻¹). `HR+VENT+TSI+ANTHRO`, the lowest in the pooled analysis, reached 323 ± 141 and 369 ± 158 mL·min⁻¹. Adding `ANTHRO` to `HR` lowered the RMSE in both cohorts (cohort G 425 → 285; cohort L 626 → 430 mL·min⁻¹).

**Pooled vs within-cohort models.** Scored on each cohort's participants, the pooled model gave an RMSE for `WR` of 185 ± 34 mL·min⁻¹ (cohort G) and 188 ± 26 mL·min⁻¹ (cohort L). Its RMSE was lower than that of the within-cohort model for 22 of the 24 configurations containing work rate in cohort G (mean difference −15 mL·min⁻¹) and for all 24 in cohort L (−22 mL·min⁻¹). For the 22 configurations without work rate, it was lower for 4 in cohort G (mean difference +30 mL·min⁻¹) and for 12 in cohort L (−13 mL·min⁻¹). Training-set sizes differed by design (§2.11): a pooled fold trained on 151,553 target seconds on average (range 150,870–154,510), a within-cohort fold on 71,438 (cohort G) or 74,261 (cohort L).

## 3.3 Agreement and peak V̇O₂ (Aim 1) [main]

Table 2 summarises agreement for `WR`, the two other carried-forward configurations, and `HR` with and without anthropometrics.

**Correlation and limits of agreement.** The repeated-measures correlation was 0.957 for `WR`, 0.953 for `WR+VENT`, 0.881 for `HR+VENT+TSI+ANTHRO` and 0.762 for `HR`, with an SD across seeds of at most 0.012 for these five configurations. The 95 % limits of agreement were −389 to 366 mL·min⁻¹ for `WR` and −401 to 390 mL·min⁻¹ for `WR+VENT`, against −846 to 729 mL·min⁻¹ for `HR+VENT+TSI+ANTHRO` and −1,021 to 993 mL·min⁻¹ for `HR`. The repeated-measures Bland–Altman bias over all seconds, around which these limits are centred, was −11, −5, −59 and −14 mL·min⁻¹, respectively.

**Bias per participant.** The mean bias across participants was close to zero for `WR` (−1 ± 110 mL·min⁻¹) and `WR+VENT` (+2 ± 119 mL·min⁻¹), and −62 ± 303 mL·min⁻¹ for `HR+VENT+TSI+ANTHRO`. The SDs show that individual participants were offset in both directions.

**Peak V̇O₂.** In the 22 ramp tests (mean measured peak V̇O₂ over the evaluated seconds 3,263 mL·min⁻¹), `WR` overestimated peak V̇O₂ by 216 ± 263 mL·min⁻¹ (absolute error 274 ± 198 mL·min⁻¹), and did so in 16 of 22 tests. `WR+VENT` overestimated it by 169 ± 280 mL·min⁻¹ (15 of 22 tests). `HR+VENT+TSI+ANTHRO` underestimated it on average (−110 ± 517 mL·min⁻¹; absolute error 421 ± 315 mL·min⁻¹), overestimating it in 7 of 22 tests.

## 3.4 Residual structure (Aim 1) [main: short; Table S10 supplement]

To describe when the errors arose, the residuals (predicted − measured) of the pooled analysis were examined for the three carried-forward configurations (Table S10).

**Runs of same-signed residuals.** For `WR`, the mean run length per visit ranged across protocols from 19 ± 12 s (Medium) to 145 ± 190 s (Ramp), and 0.40–0.69 of the seconds lay in runs longer than 60 s. `WR+VENT` was similar (19 ± 15 to 123 ± 180 s; 0.36–0.69). Runs were longer for `HR+VENT+TSI+ANTHRO`: 98 ± 166 s (Ramp) to 636 ± 849 s (PRBS), with 0.68–0.85 of the seconds in runs longer than 60 s. Run lengths were right-skewed. Medians were 15–37 s for `WR` and 30–124 s for `HR+VENT+TSI+ANTHRO`. In 15 of the 99 visits, the residual of `HR+VENT+TSI+ANTHRO` kept one sign throughout. For `WR`, this was the case in 3 visits, all ramp tests.

**Exercise phases.** For `WR` and `WR+VENT`, the mean bias in phases 1–7 lay between −49 and +37 mL·min⁻¹, and every 95 % CI included 0. No phase's share of the squared error exceeded 1.24 times its share of time. Ramp end was the exception: bias +165 [61, 269] mL·min⁻¹ for `WR` and +151 [40, 262] mL·min⁻¹ for `WR+VENT`, with error-to-time ratios of 2.72 and 2.60. Over the last 120 s before task failure, the mean `WR` residual rose from +75 to +372 mL·min⁻¹. In the first 30 s after a PRBS switch or an interval or step change, the mean `WR` residual stayed within ±55 mL·min⁻¹ at every second. For `HR+VENT+TSI+ANTHRO`, the CIs were 1.8–3.2 times wider than for `WR`. The largest phase biases were at sustained exercise above GET (−204 [−546, 138] mL·min⁻¹; ratio 2.34) and at ramp end (−168 [−382, 45] mL·min⁻¹; ratio 1.82).

## 3.5 Generalisation (Aim 2) [main; Table S11 supplement]

> **INTERIM — do not cite these numbers as final.** This section uses the interim Aim 2 output of 14 Oct 2026 (`outputs/aim2/interim/`, interim Figure 3). Aims 2a and 2c are complete. Aim 2b has Ramp, PRBS and Interval; **2b Step is still training**. The 2b numbers will change slightly when Step is added: the Step rows are added, and the counts across conditions in the last paragraph change. The Ramp, PRBS and Interval rows come from their own runs and are not expected to change **[CHECK after the final run]**. Figure 3 and Table S11 must be regenerated from the final `outputs/aim2/summary.csv`.

Aim 2 compared, within each carried-forward configuration, a model that had not seen a condition in training with one that had, on the same test seconds (Figure 3; Table S11). Differences are unseen − seen, mean [95 % CI] across participants, in mL·min⁻¹.

**Figure 3.** Generalisation to conditions absent from training (Aim 2; interim, without 2b Step). For each carried-forward configuration and test condition: the mean RMSE across participants with the condition seen (open circle) and unseen (filled circle) in training, joined by a line, and the mean difference unseen − seen with its 95 % CI (right-hand column). (A) Aim 2a, one PRBS intensity level left out (cohort G; n = 12 for Easy, 13 for Medium and Hard). (B) Aim 2b, one protocol left out (cohort L; n = 13); Step pending. In A and B, the seen model was trained on a training set of the same size. (C) Aim 2c, transfer between cohorts, all visits: model trained on the other cohort (unseen) vs the within-cohort analysis of the test cohort (seen; G → L n = 13, L → G n = 14).

**Unseen intensity (Aim 2a).** With one PRBS level left out, the RMSE of `WR` changed by +7 [−11, 24] (Easy), +45 [23, 66] (Medium) and +10 [−32, 52] (Hard). For `WR+VENT`, the differences were +19 [−10, 49], +24 [3, 44] and +13 [−6, 32]; for `HR+VENT+TSI+ANTHRO`, −21 [−88, 46], +5 [−29, 38] and +23 [−20, 66]. The CI excluded 0 for Medium with `WR` and with `WR+VENT`.

**Unseen protocol (Aim 2b, interim).** With one protocol left out, the RMSE of `WR` changed by +28 [2, 53] (Ramp), +24 [10, 39] (PRBS) and +10 [−4, 23] (Interval). For `WR+VENT`, the differences were +20 [−18, 58], +7 [−2, 16] and +15 [−5, 36]; for `HR+VENT+TSI+ANTHRO`, +10 [−61, 80], +11 [−24, 47] and +17 [−14, 49]. The CI excluded 0 for Ramp and PRBS with `WR`. Step: **[CHECK: 2b Step not yet available; still training on 14 Oct 2026]**.

**Unseen cohort (Aim 2c).** Over all visits of the test cohort, `WR` trained on the other cohort reached the RMSE of the within-cohort reference: 196 → 196 mL·min⁻¹ for G → L (0 [−15, 15]) and 194 → 200 mL·min⁻¹ for L → G (+6 [−12, 25]). For `WR+VENT`, the differences were +30 [−15, 76] (G → L) and +9 [−35, 54] (L → G). For `HR+VENT+TSI+ANTHRO`, they were +53 [−81, 186] (G → L) and +121 [31, 212] (L → G; 323 → 444 mL·min⁻¹), and the CI excluded 0 for L → G. The per-protocol breakdown is given in Table S11. The ramp is the only protocol that both cohorts performed, so only there does the transfer change the cohort without changing the protocol.

Across the 24 interim comparisons, the mean difference was positive in 22, and the CI excluded 0 in 5.

---

## Open points for co-authors

Things Claude was unsure about, choices to confirm, and missing numbers. Numbered so they can be answered by number.

1. **2b Step (missing).** Still training. When it is done, regenerate `outputs/aim2/summary.csv` and Figure 3 (`figures/aim2_generalisation.py`, final output path, not `interim/`), fill the Step rows in §3.5 and Table S11, and update the counts "positive in 22 of 24; CI excludes 0 in 5" (these become out of 27). Also check that the Ramp, PRBS and Interval rows are unchanged.
2. **Which Figure 3 file.** `outputs/paper/aim2_generalisation.png` (14 Oct, 17:33) is the **2c-only** version, built from `outputs/aim2/summary.csv`, which holds 2c rows only. The interim figure that matches this text is `outputs/paper/interim/aim2_generalisation.png`. Do not paste the wrong one.
3. **Hypotheses H1–H3** (`01_introduction.md`) are still marked "to be fixed before the results are seen", and the results now exist. The draft describes the numbers that bear on them (`WR` lowest; the other 23 configurations with WR +10 to +34; among the four lowest configurations without WR, three contain HR) but does not judge them, because the inferential tests are deferred. Decide how the hypotheses are stated and whether the results should refer to them explicitly.
4. **Two different "bias" values.** Table 2's Bias column is the participant-level mean bias (`WR` −1 ± 110). The limits of agreement are centred on the repeated-measures Bland–Altman bias over all seconds (`WR` −11), which is not in Table 2. §3.3 reports both. Consider adding a BA-bias column to Table 2 or a footnote, so that readers do not take the LoA as centred on −1.
5. **Term: "VO₂peak error" vs "peak V̇O₂".** Table 2 (generated by `figures/aim1_key_configurations.py`) uses "VO₂peak", and the methods (§2.13) use "peak V̇O₂". The text follows the methods. Pick one and change the table script if needed.
6. **Run length: mean ± SD vs median.** The methods (§2.14, §2.15) specify mean ± SD. The distributions are strongly right-skewed (SD > mean for `HR+VENT+TSI+ANTHRO` in every protocol and for `WR` in four of seven), so the dev's residual-analysis notes (ticket 09; see point 8) suggested a median and IQR. The text gives both, and Table S10 adds the median (range). If medians stay, add half a sentence to methods §2.15.
7. **Runs test in Table S10.** This draft follows the descriptive-only rule (no p-values), so Table S10 shows observed vs expected runs and z, but not p (p ≈ 0 everywhere; `runs_test.csv`). Decide whether to show z at all, or drop the runs test from the paper.
8. **Ticket 09 vs the output files.** Ticket 09 is the dev's working summary of the residual run (`.claude-local/wayfinder/aim2-and-paper/issues/09-residual-analysis-script.md`; gitignored, so co-authors may not have it). Two numbers in it don't match `traces.csv`, and the text uses the file values:
   - the `WR` mean residual in the first 30 s after a change reaches −54.5 (phase 2, offset 1 s), not "within ±45";
   - after Step increases, 60–179 s, the `WR` residual ranges from −16 to −92 (39 events), not "−44 … −71".
9. **8-s staircase in the aligned traces (looks odd, not interpreted).** After every work-rate change, the mean `WR` prediction moves in steps at offsets 1, 9, 17 and 25 s. For example, after a PRBS switch up the residual is +42, +37, +32 and +28 at these offsets, and −18 to −23 one second before each (offsets 8, 16, 24). This is probably the TCN's dilated kernel taps (dilation 8) acting on a step input. It could matter for item 6 and for the discussion of response dynamics. Worth a look by someone who knows the model.
10. **Ramp-end trace composition.** "+75 → +372 mL·min⁻¹" compares 22 ramps at −120 s with 16 ramps at −1 s (6 ramps lose their final seconds). The measured "plateau" in ticket 09 (3,033 → 3,294) has the same composition change. Decide whether to restrict to the 16 complete ramps or to state the n.
11. **ANTHRO direction differs by cohort.** In the pooled analysis, `ANTHRO` raised the RMSE of every work-rate configuration. Within cohort L it did so in 11 of 12 pairs, but within cohort G only in 7 of 12 (`A1_gies`, −8 to +32). Not in the text; add it if the discussion needs it.
12. **Pooled-vs-within comparison is confounded by training size** (≈ 2.0–2.1× the target seconds). Methods §2.11 say sizes "are reported with the results", which §3.2 does. Whether to frame this as secondary aim (iii) is a discussion choice.
13. **Peak V̇O₂ reference value.** The mean measured peak V̇O₂ (3,263 mL·min⁻¹) is over the 22 included ramps and their evaluated seconds (`vo2peak.csv`). It is not the Table 1 value (3.30 L·min⁻¹, all 27 participants, cart software). Keep it, or drop it to avoid confusion.
14. **Rounding.** Table S11 rounds the seen, unseen and Δ values independently, so a few rows look inconsistent by 1 mL·min⁻¹ (e.g. 2a Easy `WR+VENT`: 182 → 202, Δ +19, because the exact Δ is 19.5). Decide whether to show one decimal for Δ.
15. **Supplementary numbering.** S8–S11 continue the methods. The map's "supplement contents" item is still open (e.g. Bland–Altman plots, sensitivity runs `A1_*_lr5e-4_ep1`, which are **not** reported here), so the numbers may change.
16. **Sensitivity analyses missing.** The `lr5e-4 × 1` sensitivity runs (methods §2.16) exist (`outputs/runs/A1_*_lr5e-4_ep1`), but nothing about them is in this draft. **[CHECK: sensitivity results to be added — supplement]**.
17. **Pooled model per cohort as a 2c reference** (methods §2.12, supplement) isn't reported. The values are in `A1_pooled/summary/summary.csv` (cohort groups), e.g. `WR` 185 (G) and 188 (L). Add it to Table S11 if wanted.
18. **Length.** The drafts are deliberately long: write everything out in full now, and cut in a later pass (dev, 14 Oct). Add detail rather than trim it.

---

## Supplementary tables [supplement]

**Table S8.** All 46 input configurations (Aim 1), ordered by pooled RMSE.

| Configuration | RMSE, pooled | SD across seeds | MAE, pooled | Bias, pooled | r_rm, pooled | RMSE, cohort G only (n = 14) | RMSE, cohort L only (n = 13) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `WR` | 186 ± 30 | 1.0 | 154 ± 29 | −1 ± 110 | 0.957 | 194 ± 36 | 196 ± 29 |
| `WR+VENT` | 196 ± 39 | 2.6 | 163 ± 37 | 2 ± 119 | 0.953 | 206 ± 81 | 200 ± 34 |
| `WR+HR+VENT+HBDIFF` | 197 ± 43 | 2.2 | 163 ± 40 | −19 ± 115 | 0.951 | 237 ± 82 | 239 ± 86 |
| `WR+HR+TSI` | 197 ± 35 | 3.6 | 164 ± 32 | −15 ± 106 | 0.948 | 225 ± 49 | 230 ± 71 |
| `WR+VENT+HBDIFF` | 198 ± 44 | 3.2 | 164 ± 41 | −16 ± 121 | 0.952 | 212 ± 93 | 228 ± 74 |
| `WR+TSI` | 198 ± 41 | 3.2 | 164 ± 39 | −6 ± 121 | 0.952 | 208 ± 39 | 211 ± 40 |
| `WR+HR+VENT` | 200 ± 48 | 1.5 | 166 ± 47 | −7 ± 122 | 0.950 | 224 ± 81 | 231 ± 54 |
| `WR+ANTHRO` | 200 ± 52 | 3.5 | 167 ± 50 | 1 ± 126 | 0.951 | 226 ± 68 | 211 ± 67 |
| `WR+VENT+TSI` | 201 ± 42 | 3.9 | 168 ± 41 | −5 ± 126 | 0.951 | 210 ± 72 | 220 ± 51 |
| `WR+HR+VENT+TSI` | 201 ± 47 | 3.2 | 168 ± 44 | −14 ± 121 | 0.951 | 227 ± 83 | 244 ± 83 |
| `WR+HBDIFF` | 202 ± 40 | 2.8 | 167 ± 38 | −15 ± 122 | 0.951 | 217 ± 58 | 206 ± 51 |
| `WR+HR` | 204 ± 41 | 0.8 | 170 ± 41 | −13 ± 116 | 0.943 | 216 ± 51 | 214 ± 58 |
| `WR+HR+HBDIFF` | 206 ± 38 | 5.2 | 171 ± 37 | −13 ± 112 | 0.943 | 220 ± 41 | 212 ± 55 |
| `WR+TSI+ANTHRO` | 206 ± 54 | 2.7 | 174 ± 52 | −13 ± 138 | 0.951 | 212 ± 36 | 206 ± 47 |
| `WR+HR+VENT+HBDIFF+ANTHRO` | 211 ± 71 | 7.6 | 179 ± 66 | −24 ± 143 | 0.948 | 228 ± 101 | 257 ± 85 |
| `WR+VENT+HBDIFF+ANTHRO` | 211 ± 55 | 8.0 | 178 ± 52 | −10 ± 140 | 0.950 | 211 ± 83 | 248 ± 90 |
| `WR+HR+VENT+TSI+ANTHRO` | 213 ± 61 | 9.4 | 181 ± 58 | −16 ± 139 | 0.946 | 234 ± 89 | 285 ± 78 |
| `WR+HBDIFF+ANTHRO` | 215 ± 59 | 1.2 | 182 ± 56 | −27 ± 150 | 0.950 | 236 ± 61 | 213 ± 58 |
| `WR+VENT+ANTHRO` | 216 ± 61 | 2.9 | 183 ± 59 | −1 ± 148 | 0.946 | 210 ± 73 | 231 ± 80 |
| `WR+HR+HBDIFF+ANTHRO` | 216 ± 56 | 6.6 | 183 ± 52 | −25 ± 139 | 0.946 | 231 ± 89 | 241 ± 67 |
| `WR+VENT+TSI+ANTHRO` | 219 ± 61 | 4.5 | 186 ± 60 | −13 ± 155 | 0.948 | 229 ± 71 | 243 ± 80 |
| `WR+HR+VENT+ANTHRO` | 219 ± 77 | 6.8 | 187 ± 73 | −16 ± 158 | 0.947 | 220 ± 77 | 258 ± 93 |
| `WR+HR+TSI+ANTHRO` | 219 ± 52 | 6.3 | 187 ± 49 | −20 ± 145 | 0.945 | 223 ± 57 | 245 ± 65 |
| `WR+HR+ANTHRO` | 220 ± 64 | 2.6 | 187 ± 60 | −20 ± 150 | 0.945 | 212 ± 58 | 252 ± 88 |
| `HR+VENT+TSI+ANTHRO` | 362 ± 158 | 8.0 | 320 ± 148 | −62 ± 303 | 0.881 | 323 ± 141 | 369 ± 158 |
| `HR+TSI+ANTHRO` | 372 ± 174 | 13.2 | 326 ± 161 | −52 ± 304 | 0.860 | 329 ± 114 | 382 ± 146 |
| `HR+VENT+HBDIFF` | 376 ± 112 | 12.7 | 330 ± 109 | −22 ± 288 | 0.865 | 329 ± 122 | 424 ± 237 |
| `VENT+TSI` | 378 ± 157 | 3.8 | 328 ± 148 | −24 ± 314 | 0.828 | 349 ± 144 | 394 ± 175 |
| `VENT+HBDIFF` | 388 ± 169 | 7.2 | 337 ± 164 | −17 ± 328 | 0.855 | 399 ± 205 | 381 ± 164 |
| `HR+VENT+HBDIFF+ANTHRO` | 390 ± 169 | 7.5 | 350 ± 166 | −34 ± 355 | 0.893 | 315 ± 149 | 325 ± 131 |
| `HR+HBDIFF+ANTHRO` | 394 ± 190 | 14.2 | 351 ± 184 | −30 ± 352 | 0.860 | 304 ± 139 | 373 ± 175 |
| `VENT+TSI+ANTHRO` | 405 ± 159 | 6.6 | 354 ± 154 | −62 ± 332 | 0.838 | 429 ± 200 | 387 ± 141 |
| `HR+VENT+ANTHRO` | 405 ± 138 | 8.0 | 361 ± 132 | −23 ± 349 | 0.875 | 291 ± 101 | 381 ± 137 |
| `HR+HBDIFF` | 405 ± 126 | 10.6 | 350 ± 114 | −30 ± 287 | 0.819 | 384 ± 148 | 425 ± 221 |
| `VENT+HBDIFF+ANTHRO` | 406 ± 136 | 6.8 | 359 ± 135 | −35 ± 349 | 0.874 | 415 ± 178 | 355 ± 100 |
| `HR+ANTHRO` | 406 ± 176 | 9.5 | 363 ± 172 | −15 ± 360 | 0.858 | 285 ± 103 | 430 ± 184 |
| `HR+VENT+TSI` | 415 ± 179 | 16.0 | 368 ± 172 | −46 ± 367 | 0.838 | 315 ± 108 | 401 ± 235 |
| `HR+TSI` | 415 ± 153 | 11.3 | 364 ± 139 | −20 ± 330 | 0.824 | 404 ± 173 | 425 ± 227 |
| `TSI+ANTHRO` | 418 ± 159 | 16.2 | 357 ± 144 | −18 ± 283 | 0.774 | 531 ± 209 | 461 ± 225 |
| `VENT` | 431 ± 142 | 4.1 | 369 ± 136 | 1 ± 325 | 0.785 | 406 ± 167 | 487 ± 113 |
| `HBDIFF+ANTHRO` | 433 ± 199 | 7.3 | 375 ± 185 | −14 ± 351 | 0.789 | 503 ± 295 | 422 ± 226 |
| `TSI` | 436 ± 164 | 5.4 | 378 ± 155 | −29 ± 332 | 0.770 | 443 ± 118 | 427 ± 189 |
| `HBDIFF` | 452 ± 165 | 3.2 | 389 ± 152 | −22 ± 344 | 0.770 | 496 ± 167 | 457 ± 248 |
| `VENT+ANTHRO` | 454 ± 142 | 12.9 | 391 ± 134 | −27 ± 350 | 0.785 | 408 ± 129 | 446 ± 121 |
| `HR` | 481 ± 142 | 8.0 | 422 ± 136 | −8 ± 371 | 0.762 | 425 ± 119 | 626 ± 238 |
| `HR+VENT` | 483 ± 177 | 7.0 | 427 ± 173 | 3 ± 410 | 0.796 | 336 ± 121 | 653 ± 183 |

Values in mL·min⁻¹ (r_rm unitless). RMSE, MAE and bias: mean ± SD across participants of the seed-averaged values (pooled n = 27). SD across seeds: SD of the five per-seed configuration means. r_rm: mean of 5 seeds. Within-cohort columns: leave-one-subject-out within that cohort. Bias = predicted − measured.

**Table S9.** RMSE per protocol in the pooled analysis, for the Table 2 configurations.

| Configuration | Ramp (n = 22) | Easy (n = 12) | Medium (n = 13) | Hard (n = 13) | PRBS (n = 13) | Interval (n = 13) | Step (n = 13) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `WR` | 205 ± 57 | 182 ± 80 | 166 ± 34 | 189 ± 61 | 174 ± 45 | 175 ± 43 | 203 ± 52 |
| `WR+VENT` | 221 ± 54 | 187 ± 74 | 173 ± 45 | 212 ± 72 | 183 ± 54 | 181 ± 51 | 197 ± 43 |
| `HR+VENT+TSI+ANTHRO` | 376 ± 190 | 320 ± 109 | 303 ± 110 | 378 ± 182 | 410 ± 241 | 255 ± 108 | 433 ± 243 |
| `HR` | 626 ± 270 | 365 ± 121 | 364 ± 118 | 509 ± 135 | 439 ± 186 | 412 ± 147 | 606 ± 224 |
| `HR+ANTHRO` | 399 ± 177 | 278 ± 99 | 325 ± 156 | 488 ± 265 | 455 ± 191 | 382 ± 212 | 486 ± 158 |

Values in mL·min⁻¹, mean ± SD across participants. Easy, Medium and Hard are cohort G; PRBS, Interval and Step cohort L; Ramp both cohorts.

**Table S10.** Residual analyses of the pooled analysis for the three carried-forward configurations (Aim 1; methods §2.14, Table S7).

*(a) Runs of same-signed residuals, per protocol.*

| Configuration | Protocol | n | Mean run length per visit (s), mean ± SD | Median (range) (s) | Share of seconds in runs > 60 s | Runs, observed / expected (joined) | z (joined) |
|---|---|---:|---:|---:|---:|---:|---:|
| `WR` | Ramp | 22 | 145 ± 190 | 37 (11–608) | 0.66 ± 0.32 | 461 / 6,908 | −110 |
| `WR` | Easy | 12 | 57 ± 104 | 22 (13–383) | 0.55 ± 0.30 | 1,015 / 11,469 | −138 |
| `WR` | Medium | 13 | 19 ± 12 | 15 (12–58) | 0.40 ± 0.21 | 1,518 / 11,990 | −135 |
| `WR` | Hard | 13 | 36 ± 41 | 17 (12–148) | 0.41 ± 0.32 | 1,222 / 12,000 | −139 |
| `WR` | PRBS | 13 | 31 ± 26 | 21 (15–109) | 0.57 ± 0.21 | 1,027 / 11,871 | −142 |
| `WR` | Interval | 13 | 35 ± 33 | 20 (10–124) | 0.53 ± 0.27 | 1,110 / 11,770 | −141 |
| `WR` | Step | 13 | 89 ± 161 | 35 (18–619) | 0.69 ± 0.22 | 644 / 11,860 | −147 |
| `WR+VENT` | Ramp | 22 | 123 ± 180 | 39 (12–608) | 0.69 ± 0.26 | 391 / 6,811 | −111 |
| `WR+VENT` | Easy | 12 | 35 ± 40 | 19 (13–147) | 0.52 ± 0.28 | 1,136 / 11,516 | −137 |
| `WR+VENT` | Medium | 13 | 19 ± 15 | 13 (10–66) | 0.36 ± 0.25 | 1,662 / 11,947 | −133 |
| `WR+VENT` | Hard | 13 | 51 ± 60 | 22 (10–213) | 0.52 ± 0.33 | 1,048 / 11,992 | −141 |
| `WR+VENT` | PRBS | 13 | 48 ± 98 | 18 (14–372) | 0.51 ± 0.23 | 1,179 / 11,892 | −140 |
| `WR+VENT` | Interval | 13 | 35 ± 42 | 22 (11–169) | 0.53 ± 0.27 | 1,175 / 12,010 | −140 |
| `WR+VENT` | Step | 13 | 45 ± 49 | 30 (17–206) | 0.64 ± 0.17 | 780 / 11,817 | −145 |
| `HR+VENT+TSI+ANTHRO` | Ramp | 22 | 98 ± 166 | 44 (15–668) | 0.80 ± 0.15 | 343 / 6,866 | −112 |
| `HR+VENT+TSI+ANTHRO` | Easy | 12 | 317 ± 552 | 75 (17–1,916) | 0.80 ± 0.25 | 457 / 11,312 | −146 |
| `HR+VENT+TSI+ANTHRO` | Medium | 13 | 156 ± 292 | 31 (13–952) | 0.72 ± 0.22 | 742 / 11,567 | −145 |
| `HR+VENT+TSI+ANTHRO` | Hard | 13 | 634 ± 895 | 77 (15–1,920) | 0.73 ± 0.31 | 667 / 10,827 | −145 |
| `HR+VENT+TSI+ANTHRO` | PRBS | 13 | 636 ± 849 | 124 (17–1,860) | 0.82 ± 0.25 | 396 / 11,860 | −150 |
| `HR+VENT+TSI+ANTHRO` | Interval | 13 | 220 ± 518 | 30 (12–1,858) | 0.68 ± 0.23 | 805 / 11,975 | −145 |
| `HR+VENT+TSI+ANTHRO` | Step | 13 | 445 ± 663 | 74 (24–1,860) | 0.85 ± 0.18 | 391 / 12,074 | −150 |

n = participants (one visit each per protocol). Run length per visit = target seconds ÷ number of runs; summarised across participants. A per-visit mean run length equal to the visit's target seconds means that the residual kept one sign throughout. Joined runs test: runs counted within visits and summed; expected runs from the pooled signs (E = 2n₊n₋/n + 1); no residual was exactly zero. At 1 Hz the joined test departs from its null by construction (methods §2.14); it is shown for completeness, without p-values.

*(b) Bias and error share per exercise phase (primary windows).*

| Group | Phase | n participants | Events | Seconds | `WR` | `WR+VENT` | `HR+VENT+TSI+ANTHRO` |
|---|---|---:|---:|---:|---:|---:|---:|
| Pooled | 1 PRBS switch up | 27 | 947 | 28,325 | 5 [−37, 47]; 0.93 | 7 [−40, 55]; 0.96 | −54 [−175, 68]; 0.78 |
| Pooled | 2 PRBS switch down | 27 | 911 | 27,330 | −14 [−64, 35]; 1.00 | −7 [−62, 48]; 1.06 | −97 [−255, 60]; 1.07 |
| Pooled | 3 increase | 13 | 169 | 14,820 | −12 [−98, 75]; 1.17 | −23 [−111, 66]; 1.12 | −81 [−286, 125]; 1.27 |
| Pooled | 4 decrease | 13 | 169 | 21,026 | −9 [−77, 59]; 0.86 | −21 [−84, 42]; 0.83 | −24 [−146, 98]; 0.60 |
| Pooled | 5 sustained below GET | 13 | 52 | 10,107 | −49 [−148, 51]; 1.24 | −43 [−125, 39]; 0.92 | −37 [−284, 210]; 1.23 |
| Pooled | 6 sustained above GET | 13 | 13 | 2,340 | −4 [−110, 102]; 0.99 | −37 [−138, 63]; 0.88 | −204 [−546, 138]; 2.34 |
| Pooled | 7 ramp through GET | 22 | 22 | 5,280 | 7 [−60, 74]; 0.82 | 37 [−35, 110]; 0.94 | 95 [−41, 232]; 0.83 |
| Pooled | 8 ramp end | 22 | 22 | 2,619 | 165 [61, 269]; 2.72 | 151 [40, 262]; 2.60 | −168 [−382, 45]; 1.82 |
| Pooled | — other | 27 | — | 45,535 | −4 [−47, 39]; 0.92 | 3 [−45, 51]; 0.97 | −85 [−217, 47]; 1.05 |
| Cohort G | 7 ramp through GET | 9 | 9 | 2,160 | 2 [−120, 124]; 0.87 | 46 [−87, 179]; 0.96 | 217 [95, 339]; 0.85 |
| Cohort L | 7 ramp through GET | 13 | 13 | 3,120 | 10 [−81, 102]; 0.79 | 31 [−68, 130]; 0.94 | 11 [−207, 230]; 0.79 |
| Cohort G | 8 ramp end | 9 | 9 | 1,071 | 208 [22, 395]; 3.11 | 167 [15, 319]; 2.08 | −96 [−387, 196]; 1.28 |
| Cohort L | 8 ramp end | 13 | 13 | 1,548 | 135 [−6, 276]; 2.45 | 140 [−35, 314]; 3.01 | −218 [−551, 114]; 2.00 |

Cells: bias, mean [95 % t-CI] across participants, in mL·min⁻¹ (predicted − measured; each participant's mean residual over their window seconds); then the error-to-time ratio (the phase's share of squared error ÷ its share of target seconds; equal to the phase MSE ÷ the group MSE). "Other" = target seconds outside all windows. Events = events with target seconds. Phases 3–6 are cohort L only; phases 1–2 combine cohort G Easy/Medium/Hard and cohort L PRBS. Sensitivity windows (phases 1–2 until the next switch): `WR` −4 [−46, 39] and −7 [−52, 39]; `WR+VENT` 7 [−42, 56] and −4 [−53, 44]; `HR+VENT+TSI+ANTHRO` −96 [−240, 49] and −77 [−215, 61] mL·min⁻¹.

**Table S11.** Aim 2 outcomes (**interim, 14 Oct 2026; 2b Step pending**).

| Aim | Left-out condition / direction | Test visits (2c) | Configuration | n | RMSE, seen | RMSE, unseen | Δ unseen − seen [95 % CI] |
|---|---|---|---|---:|---:|---:|---:|
| 2a | Easy | — | `WR` | 12 | 186 | 193 | +7 [−11, 24] |
| 2a | Easy | — | `HR+VENT+TSI+ANTHRO` | 12 | 295 | 274 | −21 [−88, 46] |
| 2a | Easy | — | `WR+VENT` | 12 | 182 | 202 | +19 [−10, 49] |
| 2a | Medium | — | `WR` | 13 | 175 | 220 | +45 [23, 66] |
| 2a | Medium | — | `HR+VENT+TSI+ANTHRO` | 13 | 288 | 293 | +5 [−29, 38] |
| 2a | Medium | — | `WR+VENT` | 13 | 175 | 199 | +24 [3, 44] |
| 2a | Hard | — | `WR` | 13 | 193 | 203 | +10 [−32, 52] |
| 2a | Hard | — | `HR+VENT+TSI+ANTHRO` | 13 | 356 | 379 | +23 [−20, 66] |
| 2a | Hard | — | `WR+VENT` | 13 | 220 | 232 | +13 [−6, 32] |
| 2b | Ramp | — | `WR` | 13 | 216 | 243 | +28 [2, 53] |
| 2b | Ramp | — | `HR+VENT+TSI+ANTHRO` | 13 | 375 | 385 | +10 [−61, 80] |
| 2b | Ramp | — | `WR+VENT` | 13 | 229 | 249 | +20 [−18, 58] |
| 2b | PRBS | — | `WR` | 13 | 186 | 211 | +24 [10, 39] |
| 2b | PRBS | — | `HR+VENT+TSI+ANTHRO` | 13 | 401 | 412 | +11 [−24, 47] |
| 2b | PRBS | — | `WR+VENT` | 13 | 187 | 194 | +7 [−2, 16] |
| 2b | Interval | — | `WR` | 13 | 180 | 189 | +10 [−4, 23] |
| 2b | Interval | — | `HR+VENT+TSI+ANTHRO` | 13 | 286 | 303 | +17 [−14, 49] |
| 2b | Interval | — | `WR+VENT` | 13 | 183 | 198 | +15 [−5, 36] |
| 2b | Step | — | all three | **[CHECK: still training]** | | | |
| 2c | G → L | all visits | `WR` | 13 | 196 | 196 | 0 [−15, 15] |
| 2c | G → L | all visits | `HR+VENT+TSI+ANTHRO` | 13 | 369 | 422 | +53 [−81, 186] |
| 2c | G → L | all visits | `WR+VENT` | 13 | 200 | 231 | +30 [−15, 76] |
| 2c | L → G | all visits | `WR` | 14 | 194 | 200 | +6 [−12, 25] |
| 2c | L → G | all visits | `HR+VENT+TSI+ANTHRO` | 14 | 323 | 444 | +121 [31, 212] |
| 2c | L → G | all visits | `WR+VENT` | 14 | 206 | 215 | +9 [−35, 54] |
| 2c | G → L | Ramp | `WR` | 13 | 214 | 208 | −6 [−25, 13] |
| 2c | G → L | Ramp | `HR+VENT+TSI+ANTHRO` | 13 | 381 | 448 | +67 [−101, 236] |
| 2c | G → L | Ramp | `WR+VENT` | 13 | 224 | 273 | +49 [−27, 125] |
| 2c | G → L | PRBS | `WR` | 13 | 185 | 172 | −13 [−34, 7] |
| 2c | G → L | PRBS | `HR+VENT+TSI+ANTHRO` | 13 | 396 | 447 | +51 [−97, 199] |
| 2c | G → L | PRBS | `WR+VENT` | 13 | 187 | 205 | +18 [−22, 57] |
| 2c | G → L | Interval | `WR` | 13 | 179 | 199 | +19 [−10, 48] |
| 2c | G → L | Interval | `HR+VENT+TSI+ANTHRO` | 13 | 276 | 338 | +62 [−95, 220] |
| 2c | G → L | Interval | `WR+VENT` | 13 | 180 | 215 | +35 [−9, 80] |
| 2c | G → L | Step | `WR` | 13 | 206 | 206 | −1 [−29, 28] |
| 2c | G → L | Step | `HR+VENT+TSI+ANTHRO` | 13 | 423 | 453 | +30 [−108, 168] |
| 2c | G → L | Step | `WR+VENT` | 13 | 210 | 230 | +20 [−31, 70] |
| 2c | L → G | Ramp | `WR` | 9 | 244 | 227 | −17 [−55, 21] |
| 2c | L → G | Ramp | `HR+VENT+TSI+ANTHRO` | 9 | 331 | 406 | +75 [−78, 228] |
| 2c | L → G | Ramp | `WR+VENT` | 9 | 242 | 234 | −8 [−116, 100] |
| 2c | L → G | Easy | `WR` | 12 | 184 | 207 | +23 [−22, 67] |
| 2c | L → G | Easy | `HR+VENT+TSI+ANTHRO` | 12 | 293 | 430 | +137 [14, 260] |
| 2c | L → G | Easy | `WR+VENT` | 12 | 183 | 209 | +25 [−39, 90] |
| 2c | L → G | Medium | `WR` | 13 | 175 | 183 | +8 [−24, 40] |
| 2c | L → G | Medium | `HR+VENT+TSI+ANTHRO` | 13 | 272 | 406 | +134 [40, 228] |
| 2c | L → G | Medium | `WR+VENT` | 13 | 176 | 201 | +25 [−10, 60] |
| 2c | L → G | Hard | `WR` | 13 | 194 | 197 | +3 [−22, 28] |
| 2c | L → G | Hard | `HR+VENT+TSI+ANTHRO` | 13 | 357 | 462 | +104 [−16, 225] |
| 2c | L → G | Hard | `WR+VENT` | 13 | 217 | 232 | +15 [−47, 77] |

Values in mL·min⁻¹; RMSE = mean across participants of the seed-averaged RMSE; Δ = mean of the per-participant differences with a 95 % t-CI; values rounded independently. 2a/2b: seen = model trained with the condition, size-matched to the unseen model; only the left-out visit is scored. 2c: unseen = model trained on the other cohort, seen = within-cohort analysis of the test cohort (`A1_lucas` for G → L, `A1_gies` for L → G); "all visits" is the primary result (Figure 3C), and the per-protocol rows are breakdowns. The ramp is the only protocol both cohorts performed, so for the other protocols an unseen cohort is confounded with an unseen protocol.

---

## Sources

All paths are relative to the repository root; values were read with awk on 14 Oct 2026.

| Section / item | File(s) |
|---|---|
| §3.1–3.3, Table S8 | `outputs/runs/A1_pooled/summary/summary.csv` (group `all`, and `cohort=Gies`/`cohort=Lucas` for the pooled model per cohort); `outputs/runs/A1_gies/…` and `outputs/runs/A1_lucas/…/summary.csv` (group `all`) |
| Table 2 | `outputs/paper/aim1_key_configurations.md` (copied verbatim; script `figures/aim1_key_configurations.py`) |
| Figure 2 | `outputs/paper/aim1_dot_plot.png` (script `figures/aim1_dot_plot.py`) |
| Participant counts (§3.1: lower than `WR` in k of 27) | `outputs/runs/A1_pooled/summary/visit_metrics_seedavg.csv` (participant RMSE = mean of visit RMSE) |
| Training-set sizes (§3.2) | `outputs/runs/A1_{pooled,gies,lucas}/summary/folds.csv` (`n_train_targets`) |
| Peak V̇O₂ counts and measured mean (§3.3) | `outputs/runs/A1_pooled/summary/vo2peak.csv` |
| Table S9 | `outputs/runs/A1_pooled/summary/summary.csv` (groups `protocol=…`) |
| §3.4, Table S10 | `outputs/residuals/A1_pooled/`: `run_length.csv`, `visit_runs.csv` (medians, ranges, single-run visits), `runs_test.csv`, `phases.csv` (labelling `primary`; sensitivity `prbs_whole_run`), `traces.csv` (ramp-end and post-change residuals) |
| §3.5, Table S11 | `outputs/aim2/interim/summary.csv` |
| Figure 3 | `outputs/paper/interim/aim2_generalisation.png` (script `figures/aim2_generalisation.py`) |
