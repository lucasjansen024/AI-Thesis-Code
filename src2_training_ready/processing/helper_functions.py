import pandas as pd
import numpy as np

from ..data_types import NIRSFileKeys, Participant, Visit


class HelperFunctions:
    MANUAL_NIRS_LIMITS = {
        (Participant.P01, Visit.Ramp): {
            NIRSFileKeys.TSI: {"min": 38, "max": 66.3},
            
        },
        (Participant.P02, Visit.Ramp): {
            NIRSFileKeys.TSI: {"min": 59.0, "max": 76.5},
        },
    }

    MANUAL_FEATURE_LIMITS = {}

    

    @staticmethod
    def reject_outliers_pre_local_mean(df: pd.DataFrame, column: str, threshold=3, window=5) -> pd.DataFrame:
        means = df[column].rolling(window=window).mean()
        stds = df[column].rolling(window=window).std()
        df['z_scores'] = 0.0

        for i in range(len(df)):
            if i < window:
                continue
            mean = means[i - 1]
            std = stds[i - 1]
            z_score = np.abs((df[column][i] - mean) / std)
            df.loc[i, 'z_scores'] = z_score

        df.loc[df['z_scores'] >= threshold, column] = np.nan
        df.drop(columns=['z_scores'], inplace=True)

        return df

    @staticmethod
    def median_filter(df: pd.DataFrame, column: str, window=5, center=True) -> pd.DataFrame:
        df[column] = df[column].rolling(window=window, center=center).median()
        return df

    @staticmethod
    def downsample(df: pd.DataFrame, Hz=10) -> pd.DataFrame:
        df.index = pd.timedelta_range(start='0s', periods=len(df), freq=f'{1000//Hz}ms')
        df = df.resample('1s').mean()
        df.index = (df.index.total_seconds()).astype(int)
        return df

    @staticmethod
    def get_visit_key(visit_data):
        return visit_data.participant, visit_data.visit

    @staticmethod
    def get_manual_nirs_limits(visit_data, NIRSkey):
        visit_key = HelperFunctions.get_visit_key(visit_data)
        visit_overrides = HelperFunctions.MANUAL_NIRS_LIMITS.get(visit_key, {})
        key_overrides = visit_overrides.get(NIRSkey, {})
        manual_min = key_overrides.get("min", None)
        manual_max = key_overrides.get("max", None)
        return manual_min, manual_max

    @staticmethod
    def get_manual_feature_limits(visit_data, feature_name: str):
        visit_key = HelperFunctions.get_visit_key(visit_data)
        visit_overrides = HelperFunctions.MANUAL_FEATURE_LIMITS.get(visit_key, {})
        key_overrides = visit_overrides.get(feature_name, {})
        manual_min = key_overrides.get("min", None)
        manual_max = key_overrides.get("max", None)
        return manual_min, manual_max

    @staticmethod
    def normalize_NIRS(
        Oxy_df: pd.DataFrame,
        Occlusion_df: pd.DataFrame,
        NIRSkey,
        manual_min=None,
        manual_max=None,
        use_percentiles=True,
        clip=True,
    ) -> pd.Series:
        oxy = pd.to_numeric(Oxy_df[NIRSkey], errors="coerce")
        occ = pd.to_numeric(Occlusion_df[NIRSkey], errors="coerce").dropna()

        if occ.empty:
            raise ValueError(f"Normalization failed for {NIRSkey}: occlusion series is empty.")

        if manual_min is None:
            occ_min = occ.quantile(0.05) if use_percentiles else occ.min()
        else:
            occ_min = manual_min

        if manual_max is None:
            occ_max = occ.quantile(0.95) if use_percentiles else occ.max()
        else:
            occ_max = manual_max

        if pd.isna(occ_min) or pd.isna(occ_max):
            raise ValueError(f"Normalization failed for {NIRSkey}: occ_min or occ_max is NaN.")

        if occ_max <= occ_min:
            raise ValueError(
                f"Normalization failed for {NIRSkey}: occ_max <= occ_min ({occ_max} <= {occ_min})."
            )

        norm = (oxy - occ_min) / (occ_max - occ_min)

        if clip:
            norm = norm.clip(0, 1)

        return norm
    @staticmethod
    def normalize_NIRS_for_visit(
        visit_data,
        Oxy_df: pd.DataFrame,
        Occlusion_df: pd.DataFrame,
        NIRSkey
    ) -> pd.Series:
        manual_min, manual_max = HelperFunctions.get_manual_nirs_limits(visit_data, NIRSkey)
        return HelperFunctions.normalize_NIRS(
            Oxy_df=Oxy_df,
            Occlusion_df=Occlusion_df,
            NIRSkey=NIRSkey,
            manual_min=manual_min,
            manual_max=manual_max
        )

    @staticmethod
    def normalize_feature_series_for_visit(
        visit_data,
        series: pd.Series,
        feature_name: str
    ) -> pd.Series:
        manual_min, manual_max = HelperFunctions.get_manual_feature_limits(visit_data, feature_name)

        feature_min = series.min() if manual_min is None else manual_min
        feature_max = series.max() if manual_max is None else manual_max

        if pd.isna(feature_min) or pd.isna(feature_max):
            raise ValueError(f"Feature normalization failed for {feature_name}: min or max is NaN.")

        if feature_max == feature_min:
            raise ValueError(f"Feature normalization failed for {feature_name}: max == min ({feature_min}).")

        return (series - feature_min) / (feature_max - feature_min)