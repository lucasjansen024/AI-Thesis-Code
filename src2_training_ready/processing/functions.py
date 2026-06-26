from typing import List

from ..data_types import PreparedVisitData, File, FitFileKeys, VEBxBFileKeys, NIRSFileKeys
from .helper_functions import HelperFunctions
from .function import VO2ProcessingFunction, VEProcessingFunction, HRProcessingFunction, WRProcessingFunction, NIRSProcessingFunction, FITProcessingFunction
import pandas as pd


class Process_WR(WRProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        WR_df = visit_data.data[File.WR]
        return WR_df[['WR']]

    def required_files(self) -> List[File]:
        return [File.WR]

# region VO2

class Process_VO2_Raw(VO2ProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        VO2_df = visit_data.data[File.VO2BxB]
        return VO2_df[['VO2', 't']]

    def required_files(self) -> List[File]:
        return [File.VO2BxB]

class Process_VO2_OR_median(VO2ProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        VO2_df = visit_data.data[File.VO2BxB]
        VO2_df = HelperFunctions.reject_outliers_pre_local_mean(VO2_df, 'VO2', 3, 5)
        VO2_df = HelperFunctions.median_filter(VO2_df, 'VO2', 5)
        return VO2_df[['VO2', 't']]

    def required_files(self) -> List[File]:
        return [File.VO2BxB]


class Process_VO2_OR_median_norm(VO2ProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        VO2_df = visit_data.data[File.VO2BxB]
        VO2_df = HelperFunctions.reject_outliers_pre_local_mean(VO2_df, 'VO2', 3, 5)
        VO2_df = HelperFunctions.median_filter(VO2_df, 'VO2', 5)

        Fit_df = visit_data.data[File.Fit]
        VO2_df['VO2'] = VO2_df['VO2'] / Fit_df[FitFileKeys.weight].mean()

        return VO2_df[['VO2', 't']]

    def required_files(self) -> List[File]:
        return [File.VO2BxB, File.Fit]

# endregion

# region HR

class Process_HR(HRProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        HR_df = visit_data.data[File.HR]
        return HR_df[['HR']]

    def required_files(self) -> List[File]:
        return [File.HR]


class Process_HR_HRR(HRProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        HR_df = visit_data.data[File.HR]
        return HR_df

    def required_files(self) -> List[File]:
        return [File.HR]

class Process_HRR(HRProcessingFunction):

    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        HR_df = visit_data.data[File.HR]
        return HR_df[['HRR']]

    def required_files(self) -> List[File]:
        return [File.HR]

# endregion

# region VE

class Process_VE_OR(VEProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        VEBxB_df = visit_data.data[File.VEBxB]
        VEBxB_df = HelperFunctions.reject_outliers_pre_local_mean(VEBxB_df, VEBxBFileKeys.VEBxB, 3, 5)
        return VEBxB_df[[VEBxBFileKeys.VEBxB, 't']]

    def required_files(self) -> List[File]:
        return [File.VEBxB]

class Process_VE_BR_OR(VEProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        VEBxB_df = visit_data.data[File.VEBxB]
        VEBxB_df = HelperFunctions.reject_outliers_pre_local_mean(VEBxB_df, VEBxBFileKeys.VEBxB, 3, 5)
        VEBxB_df = HelperFunctions.reject_outliers_pre_local_mean(VEBxB_df, VEBxBFileKeys.BRBxB, 3, 5)
        return VEBxB_df[[VEBxBFileKeys.VEBxB, VEBxBFileKeys.BRBxB, 't']]

    def required_files(self) -> List[File]:
        return [File.VEBxB]

class Process_VE_OR_median(VEProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        VEBxB_df = visit_data.data[File.VEBxB]
        VEBxB_df = HelperFunctions.reject_outliers_pre_local_mean(VEBxB_df, VEBxBFileKeys.VEBxB, 3, 5)
        VEBxB_df = HelperFunctions.median_filter(VEBxB_df, VEBxBFileKeys.VEBxB, 5, center=False)
        return VEBxB_df[[VEBxBFileKeys.VEBxB, 't']]

    def required_files(self) -> List[File]:
        return [File.VEBxB]

class Process_VE_BR_OR_median(VEProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        VEBxB_df = visit_data.data[File.VEBxB]
        VEBxB_df = HelperFunctions.reject_outliers_pre_local_mean(VEBxB_df, VEBxBFileKeys.VEBxB, 3, 5)
        VEBxB_df = HelperFunctions.median_filter(VEBxB_df, VEBxBFileKeys.VEBxB, 5, center=False)
        VEBxB_df = HelperFunctions.reject_outliers_pre_local_mean(VEBxB_df, VEBxBFileKeys.BRBxB, 3, 5)
        VEBxB_df = HelperFunctions.median_filter(VEBxB_df, VEBxBFileKeys.BRBxB, 5, center=False)
        return VEBxB_df[[VEBxBFileKeys.VEBxB, VEBxBFileKeys.BRBxB, 't']]

    def required_files(self) -> List[File]:
        return [File.VEBxB]


class Process_VE_measured_OR(VEProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        VO2BxB_df = visit_data.data[File.VO2BxB]
        VO2BxB_df = HelperFunctions.reject_outliers_pre_local_mean(VO2BxB_df, 'VE', 3, 5) # Apparently this was never in there
        return VO2BxB_df[['VE', 't']]

    def required_files(self) -> List[File]:
        return [File.VO2BxB]

class Process_VE_measured_OR_median(VEProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        VO2BxB_df = visit_data.data[File.VO2BxB]
        VO2BxB_df = HelperFunctions.reject_outliers_pre_local_mean(VO2BxB_df, 'VE', 3, 5)
        VO2BxB_df = HelperFunctions.median_filter(VO2BxB_df, 'VE', 5, center=False)
        return VO2BxB_df[['VE', 't']]

    def required_files(self) -> List[File]:
        return [File.VO2BxB]
# endregion VE

# region NIRS
class Process_TSI_norm(NIRSProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Oxy_df = visit_data.data[File.Oxy].copy()
        Occlusion_df = visit_data.data[File.Occlusion].copy()

        manual_min, manual_max = HelperFunctions.get_manual_nirs_limits(
            visit_data, NIRSFileKeys.TSI
        )

        print("\nTSI DEBUG")
        print("participant:", visit_data.participant)
        print("visit:", visit_data.visit)
        print("manual_min:", manual_min)
        print("manual_max:", manual_max)
        print("oxy raw min/max:", Oxy_df[NIRSFileKeys.TSI].min(), Oxy_df[NIRSFileKeys.TSI].max())
        print("occ raw min/max:", Occlusion_df[NIRSFileKeys.TSI].min(), Occlusion_df[NIRSFileKeys.TSI].max())

        Oxy_df[NIRSFileKeys.TSI] = HelperFunctions.normalize_NIRS_for_visit(
            visit_data, Oxy_df, Occlusion_df, NIRSFileKeys.TSI
        )

        print("oxy norm min/max:", Oxy_df[NIRSFileKeys.TSI].min(), Oxy_df[NIRSFileKeys.TSI].max())

        return Oxy_df[[NIRSFileKeys.TSI]]
class Process_TSI(NIRSProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Oxy_df = visit_data.data[File.Oxy].copy()
        return Oxy_df[[NIRSFileKeys.TSI]]


class Process_Hb_diff(NIRSProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        
        Oxy_df = visit_data.data[File.Oxy].copy()
        Hbdiff1 = Oxy_df[NIRSFileKeys.O2Hb1] - Oxy_df[NIRSFileKeys.HHb1]
        Hbdiff2 = Oxy_df[NIRSFileKeys.O2Hb2] - Oxy_df[NIRSFileKeys.HHb2]
        Hbdiff3 = Oxy_df[NIRSFileKeys.O2Hb3] - Oxy_df[NIRSFileKeys.HHb3]
        Oxy_df['Hb_diff'] = (Hbdiff1 + Hbdiff2 + Hbdiff3) / 3
        return Oxy_df[['Hb_diff']]


class Process_Hb_diff_norm(NIRSProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        print(">>> Process_Hb_diff_norm IS RUNNING for", visit_data.participant, visit_data.visit)
        print("RUNNING FOR:", visit_data.participant, visit_data.visit)
        print("MANUAL LIMITS:", HelperFunctions.get_manual_feature_limits(visit_data, "Hb_diff_avg"))
        Oxy_df = visit_data.data[File.Oxy].copy()
        Hbdiff1 = Oxy_df[NIRSFileKeys.O2Hb1] - Oxy_df[NIRSFileKeys.HHb1]
        Hbdiff2 = Oxy_df[NIRSFileKeys.O2Hb2] - Oxy_df[NIRSFileKeys.HHb2]
        Hbdiff3 = Oxy_df[NIRSFileKeys.O2Hb3] - Oxy_df[NIRSFileKeys.HHb3]
        Oxy_df['Hb_diff'] = (Hbdiff1 + Hbdiff2 + Hbdiff3) / 3

        print("before norm min/max:", Oxy_df['Hb_diff'].min(), Oxy_df['Hb_diff'].max())

        Oxy_df['Hb_diff'] = HelperFunctions.normalize_feature_series_for_visit(
            visit_data,
            Oxy_df['Hb_diff'],
            "Hb_diff_avg"
        )

        print("after norm min/max:", Oxy_df['Hb_diff'].min(), Oxy_df['Hb_diff'].max())

        return Oxy_df[['Hb_diff']]


class Process_Hb_diff3_norm(NIRSProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Oxy_df = visit_data.data[File.Oxy].copy()

        Oxy_df['Hb_diff'] = Oxy_df[NIRSFileKeys.O2Hb3] - Oxy_df[NIRSFileKeys.HHb3]
        Oxy_df['Hb_diff'] = HelperFunctions.normalize_feature_series_for_visit(
            visit_data,
            Oxy_df['Hb_diff'],
            "Hb_diff_3"
        )

        return Oxy_df[['Hb_diff']]


class Process_Hb_diff3_occ_norm(NIRSProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Oxy_df = visit_data.data[File.Oxy].copy()
        Occlusion_df = visit_data.data[File.Occlusion].copy()

        Oxy_df['Hb_diff'] = Oxy_df[NIRSFileKeys.O2Hb3] - Oxy_df[NIRSFileKeys.HHb3]
        Occlusion_df['Hb_diff'] = Occlusion_df[NIRSFileKeys.O2Hb3] - Occlusion_df[NIRSFileKeys.HHb3]

        Oxy_df['Hb_diff'] = HelperFunctions.normalize_NIRS_for_visit(
            visit_data, Oxy_df, Occlusion_df, 'Hb_diff'
        )

        return Oxy_df[['Hb_diff']]


class Process_HHb1_norm(NIRSProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Oxy_df = visit_data.data[File.Oxy].copy()
        Occlusion_df = visit_data.data[File.Occlusion].copy()

        Oxy_df[NIRSFileKeys.HHb1] = HelperFunctions.normalize_NIRS_for_visit(
            visit_data, Oxy_df, Occlusion_df, NIRSFileKeys.HHb1
        )
        return Oxy_df[[NIRSFileKeys.HHb1]]


class Process_HHb2_norm(NIRSProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Oxy_df = visit_data.data[File.Oxy].copy()
        Occlusion_df = visit_data.data[File.Occlusion].copy()

        Oxy_df[NIRSFileKeys.HHb2] = HelperFunctions.normalize_NIRS_for_visit(
            visit_data, Oxy_df, Occlusion_df, NIRSFileKeys.HHb2
        )
        return Oxy_df[[NIRSFileKeys.HHb2]]


class Process_HHb3_norm(NIRSProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Oxy_df = visit_data.data[File.Oxy].copy()
        Occlusion_df = visit_data.data[File.Occlusion].copy()

        Oxy_df[NIRSFileKeys.HHb3] = HelperFunctions.normalize_NIRS_for_visit(
            visit_data, Oxy_df, Occlusion_df, NIRSFileKeys.HHb3
        )
        return Oxy_df[[NIRSFileKeys.HHb3]]


class Process_HHb_avg(NIRSProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Oxy_df = visit_data.data[File.Oxy].copy()
        Occlusion_df = visit_data.data[File.Occlusion].copy()

        Oxy_df[NIRSFileKeys.HHb1] = HelperFunctions.normalize_NIRS_for_visit(
            visit_data, Oxy_df, Occlusion_df, NIRSFileKeys.HHb1
        )
        Oxy_df[NIRSFileKeys.HHb2] = HelperFunctions.normalize_NIRS_for_visit(
            visit_data, Oxy_df, Occlusion_df, NIRSFileKeys.HHb2
        )
        Oxy_df[NIRSFileKeys.HHb3] = HelperFunctions.normalize_NIRS_for_visit(
            visit_data, Oxy_df, Occlusion_df, NIRSFileKeys.HHb3
        )

        Oxy_df['HHb_avg'] = (
            Oxy_df[NIRSFileKeys.HHb1] +
            Oxy_df[NIRSFileKeys.HHb2] +
            Oxy_df[NIRSFileKeys.HHb3]
        ) / 3

        return Oxy_df[['HHb_avg']]
# endregion

# region Fit

class Process_gender(FITProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Fit_df = visit_data.data[File.Fit]
        return Fit_df[[FitFileKeys.gender]]

class Process_smm(FITProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Fit_df = visit_data.data[File.Fit]
        return Fit_df[[FitFileKeys.muscle_mass]]

class Process_weight(FITProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Fit_df = visit_data.data[File.Fit]
        return Fit_df[[FitFileKeys.weight]]

class Process_gender_smm(FITProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Fit_df = visit_data.data[File.Fit]
        return Fit_df[[FitFileKeys.gender, FitFileKeys.muscle_mass]]

class Process_gender_weight(FITProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Fit_df = visit_data.data[File.Fit]
        return Fit_df[[FitFileKeys.weight, FitFileKeys.gender]]

class Process_bf_smm(FITProcessingFunction):
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        Fit_df = visit_data.data[File.Fit]
        return Fit_df[[FitFileKeys.body_fat, FitFileKeys.muscle_mass]]


# endregion