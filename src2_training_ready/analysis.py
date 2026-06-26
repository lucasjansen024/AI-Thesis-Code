import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats

from .constants import CUT_OFF_FREQUENCY, WINDOW_LENGTH
from .data_types import BlandAltman, BlandAltmanResult, RMCorrResult, PearsonResult, ResidualRunsTestResult

class Analysis:

    @staticmethod
    def residual_runs_test(residuals, alpha: float = 0.05) -> ResidualRunsTestResult:
        residuals = pd.Series(residuals).dropna().to_numpy(dtype=float)
        signs = residuals[residuals != 0] > 0

        n = int(len(signs))
        n_positive = int(np.sum(signs))
        n_negative = int(n - n_positive)

        if n < 2:
            return ResidualRunsTestResult(
                n=n,
                n_positive=n_positive,
                n_negative=n_negative,
                runs=None,
                expected_runs=None,
                std_runs=None,
                z=None,
                p=None,
                systematic=None,
                status='not_enough_observations'
            )

        runs = int(1 + np.sum(signs[1:] != signs[:-1]))

        if n_positive == 0 or n_negative == 0:
            return ResidualRunsTestResult(
                n=n,
                n_positive=n_positive,
                n_negative=n_negative,
                runs=runs,
                expected_runs=None,
                std_runs=None,
                z=None,
                p=None,
                systematic=True,
                status='only_one_residual_sign'
            )

        expected_runs = 1 + (2 * n_positive * n_negative) / n
        variance_runs = (
            2 * n_positive * n_negative * (2 * n_positive * n_negative - n)
            / (n ** 2 * (n - 1))
        )

        if variance_runs <= 0:
            return ResidualRunsTestResult(
                n=n,
                n_positive=n_positive,
                n_negative=n_negative,
                runs=runs,
                expected_runs=float(expected_runs),
                std_runs=None,
                z=None,
                p=None,
                systematic=None,
                status='zero_run_variance'
            )

        std_runs = float(np.sqrt(variance_runs))
        z = float((runs - expected_runs) / std_runs)
        p_value = float(2 * stats.norm.sf(abs(z)))

        return ResidualRunsTestResult(
            n=n,
            n_positive=n_positive,
            n_negative=n_negative,
            runs=runs,
            expected_runs=float(expected_runs),
            std_runs=std_runs,
            z=z,
            p=p_value,
            systematic=bool(p_value < alpha),
            status='ok'
        )

    @staticmethod
    def get_harmonics(signal, cut_off=CUT_OFF_FREQUENCY):
        base_freq = 1 / len(signal)
        harmonics = math.floor(cut_off / base_freq)
        return harmonics

    @staticmethod
    def get_fourier_coefficients(
            signal,
            harmonics: int = None,
            cut_off=CUT_OFF_FREQUENCY,
    ):
        if harmonics is None:
            harmonics = Analysis.get_harmonics(signal)

        time = len(signal)
        coefficients = []
        for h in range(0, harmonics + 1):
            ASUM = 0
            BSUM = 0
            for n in range(time):
                e = 2 * np.pi * h * n / time
                A_val = np.cos(e) * signal[n]
                B_val = np.sin(e) * signal[n]
                ASUM += A_val
                BSUM += B_val
            An = 1 / time * ASUM
            Bn = 1 / time * BSUM
            AMP = np.sqrt(An ** 2 + Bn ** 2)
            # angle = np.arctan2(-Bn, An) * 180 / np.pi
            # duration = time / h if h != 0 else 0
            # frequency = 1 / duration if duration != 0 else 0
            coefficients.append((An, Bn, AMP))
        return coefficients

    @staticmethod
    def reconstruct_original_data(fourier_coeffs, time):
        reconstructed_data = []
        f1 = 1 / time
        A0 = 0
        for t in range(1, time + 1):
            harmonic_sum = 0
            for h, (An, Bn, _) in enumerate(fourier_coeffs):
                if h == 0:
                    A0 = An
                    continue

                cos = An * np.cos(2 * np.pi * h * f1 * t)
                sin = Bn * np.sin(2 * np.pi * h * f1 * t)
                harmonic_sum += cos + sin
            value = A0 + 2 * harmonic_sum
            reconstructed_data.append(value)
        return reconstructed_data

    @staticmethod
    def mng(WR, VO2, cut_off=CUT_OFF_FREQUENCY):
        harmonics = Analysis.get_harmonics(WR, cut_off=cut_off)

        # Calculate coefficients for vo2 and workrates
        VO2_coeffs = Analysis.get_fourier_coefficients(VO2, harmonics)
        WR_coeffs = Analysis.get_fourier_coefficients(WR, harmonics)

        # Calculate system gain at each harmonic
        # Skip A0, B0 and start calculating system gain from harmonic 1
        gAmps = []
        for ((VO2_A, VO2_B, VO2_amp), (WR_A, WR_B, WR_amp)) in zip(VO2_coeffs[1:], WR_coeffs[1:]):
            gAmps.append(VO2_amp / WR_amp)

        # Calculate mean of the 2nd to the last harmonic
        mean_gAmp = np.mean(gAmps[1:])

        # Return the proportion of the mean to the first harmonic in %
        return mean_gAmp / gAmps[0] * 100

    @staticmethod
    def get_fourier_reconstructed_signal(signal, cut_off=CUT_OFF_FREQUENCY):
        harmonics = Analysis.get_harmonics(signal, cut_off)
        coeffs = Analysis.get_fourier_coefficients(signal, harmonics=harmonics)
        reconstructed = Analysis.reconstruct_original_data(coeffs, len(signal))
        return reconstructed

    @staticmethod
    def mng_window(
            WR,
            VO2,
            window_length=WINDOW_LENGTH,
            step=1,
            cut_off=CUT_OFF_FREQUENCY
    ):
        windows = len(WR) - window_length + 1
        mngs = []
        for i in range(0, windows, step):
            WR_window = WR[i:i + window_length].array
            VO2_window = VO2[i:i + window_length].array

            mngs.append(Analysis.mng(WR_window, VO2_window, cut_off=cut_off))
        return mngs


    @staticmethod
    def bland_altman(measurements, estimations) -> BlandAltman:
        measurements = np.asarray(measurements)
        estimations = np.asarray(estimations)
        mean = np.mean([measurements, estimations], axis=0)
        diff = estimations - measurements  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)

        return BlandAltman(mean, diff, md, sd)

    @staticmethod
    def bland_altman_global(
            df: pd.DataFrame,
            x='Measured',
            y='Predicted',
    ) -> BlandAltmanResult:
        df['variables'] = df[y] - df[x]
        # calculate number of observations per participant
        obsv = Analysis._observations(df)

        # calcualte MS between participant and within participant values from One Way ANOVA
        MS_between, MS_within = Analysis._ANOVA_MS(df)

        # calculate SD using Repeat Measures Bland-Altman formula
        SD = Analysis._RBA_SD(obsv, MS_between, MS_within)

        # Calculate bias and limits of agreement
        bias, LOA_L, LOA_U = Analysis._RBA_values(df, SD)

        # Perform common sense testing to ensure 95% of the values align with expected 95% CI of LOA upper and lower
        commonSense = Analysis._commonSenseTesting(df, LOA_L, LOA_U)

        # # Calculate total mean and sd
        # diff = df['Measured'] - df['Predicted']
        # md = np.mean(diff)
        # sd = np.std(diff, axis=0)

        print(f"Mean Bias: {bias}, SD:{SD}, Common Sense: {commonSense}")

        return BlandAltmanResult(bias, SD)

    @staticmethod
    def rm_corr(df, x='Measured', y='Predicted'):
        required_cols = ['Participant', x, y]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"{col} column not found. Available columns: {list(df.columns)}")

        work_df = df.loc[:, required_cols].copy()
        work_df['Participant'] = work_df['Participant'].astype(str)
        work_df[x] = pd.to_numeric(work_df[x], errors='coerce')
        work_df[y] = pd.to_numeric(work_df[y], errors='coerce')
        work_df = work_df.dropna().reset_index(drop=True)

        # Alleen subjects met minstens 2 observaties houden
        counts = work_df.groupby('Participant').size()
        valid_subjects = counts[counts >= 2].index
        work_df = work_df[work_df['Participant'].isin(valid_subjects)].copy()

        n_subjects = work_df['Participant'].nunique()
        if n_subjects < 3:
            raise ValueError("rm_corr requires at least 3 participants with >= 2 observations each.")

        # Within-subject centeren
        work_df['_x_c'] = work_df[x] - work_df.groupby('Participant')[x].transform('mean')
        work_df['_y_c'] = work_df[y] - work_df.groupby('Participant')[y].transform('mean')

        xv = work_df['_x_c'].to_numpy(dtype=float)
        yv = work_df['_y_c'].to_numpy(dtype=float)

        r = np.corrcoef(xv, yv)[0, 1]

        n = len(work_df)
        dof = n - n_subjects - 1

        if not np.isfinite(r):
            p = np.nan
        elif np.isclose(abs(r), 1.0):
            p = 0.0
        elif dof <= 0:
            p = np.nan
        else:
            t = r * np.sqrt(dof / (1 - r**2))
            p = 2 * stats.t.sf(np.abs(t), dof)

        return RMCorrResult(
            r=float(r),
            p=float(p) if np.isfinite(p) else np.nan
        )


    @staticmethod
    def pearson(
            df: pd.DataFrame,
            x='Measured',
            y='Predicted',
    ) -> PearsonResult:
        pearson = stats.pearsonr(df[x], df[y])
        return PearsonResult(pearson.statistic, pearson.pvalue)

    @staticmethod
    def _observations(data):  # Identify how many ovservation were obtained for each participant
        participants = pd.unique(data['Participant'])
        obsv = np.zeros(len(participants), dtype='int64')
        for count, p in enumerate(participants):
            obsv[count] = np.sum(data['Participant'].str.count(p))

        return obsv

    @staticmethod
    def _ANOVA_MS(data):
        model = ols('variables ~ Participant', data=data).fit()
        anova_result = sm.stats.anova_lm(model, typ=2)
        MS_between = anova_result['sum_sq']['Participant'] / anova_result['df']['Participant']
        MS_within = anova_result['sum_sq']['Residual'] / anova_result['df']['Residual']

        return MS_between, MS_within

    @staticmethod
    ##Calculate Variance and SD from ANOVA results
    def _RBA_SD(obsv, MS_between, MS_within):
        diff_accrossSubjects = MS_between - MS_within
        diff_withinSubject = MS_within

        # formula is ((sum(mi)^2) - sum(mi^2)) / (n - 1) * sum(mi)
        # where m is the number of observations for participant i
        # see appendix of published manuscript for indepth explanation of these steps below : WADE, L., NEEDHAM, L., EVANS, M., MCGUIGAN, P., COLYER, S., COSKER, D. & BILZON, J. 2023. Examination of 2D frontal and sagittal markerless motion capture: Implications for markerless applications. PLOS ONE, 18, e0293917.)
        num_observations = ((np.square(obsv.sum())) - np.sum(np.square(obsv))) / ((len(obsv) - 1) * obsv.sum())

        varianceHetro = diff_accrossSubjects / num_observations

        totalVariance = varianceHetro + diff_withinSubject

        SD = np.sqrt(totalVariance)

        return (SD)

    @staticmethod
    def _RBA_values(data, SD):
        bias = data['variables'].mean()
        LOA_L = bias - (SD * 1.96)
        LOA_U = bias + (SD * 1.96)

        return bias, LOA_L, LOA_U

    @staticmethod
    # Find out how many SD are outside the LOA
    def _commonSenseTesting(data, LOA_L, LOA_U):
        commonSense = (data['variables'] > LOA_L) & (data['variables'] < LOA_U)
        length = len(data['variables'])
        commonSense = sum(commonSense == True) / length

        return commonSense