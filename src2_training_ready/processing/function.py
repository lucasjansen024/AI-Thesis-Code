from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from ..constants import TOTAL_TIME
from ..data_types import PreparedVisitData, File
from .helper_functions import HelperFunctions
import pandas as pd


class ProcessingFunctionType(Enum):
    VO2 = -3
    WR = -2
    HR = -1
    VE = 0
    NIRS = 1
    FIT = 2


class ProcessingFunction(ABC):

    def __eq__(self, other):
        if isinstance(other, ProcessingFunction):
            return self.__class__.__name__ == other.__class__.__name__
        else:
            raise TypeError(f"Unsupported type for equality check: {type(other)}")

    @property
    @abstractmethod
    def type(self) -> ProcessingFunctionType:
        pass

    def apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        df = self._apply(visit_data)

        if self.type == ProcessingFunctionType.NIRS:
            df = HelperFunctions.downsample(df)

        if self.type == ProcessingFunctionType.FIT:
            df.columns = [f'{File.Fit}_{col}' for col in df.columns]
            ind = df.index.repeat(TOTAL_TIME)
            df = df.loc[ind].reset_index(drop=True)

        return df

    @abstractmethod
    def _apply(self, visit_data: PreparedVisitData) -> pd.DataFrame:
        pass

    @abstractmethod
    def required_files(self) -> List[File]:
        pass


class NIRSProcessingFunction(ProcessingFunction):
    @property
    def type(self) -> ProcessingFunctionType:
        return ProcessingFunctionType.NIRS

    def required_files(self) -> List[File]:
        return [File.Oxy, File.Occlusion]


class FITProcessingFunction(ProcessingFunction):
    @property
    def type(self) -> ProcessingFunctionType:
        return ProcessingFunctionType.FIT

    def required_files(self) -> List[File]:
        return [File.Fit]


class VO2ProcessingFunction(ProcessingFunction):
    @property
    def type(self) -> ProcessingFunctionType:
        return ProcessingFunctionType.VO2


class VEProcessingFunction(ProcessingFunction):
    @property
    def type(self) -> ProcessingFunctionType:
        return ProcessingFunctionType.VE


class WRProcessingFunction(ProcessingFunction):
    @property
    def type(self) -> ProcessingFunctionType:
        return ProcessingFunctionType.WR


class HRProcessingFunction(ProcessingFunction):
    @property
    def type(self) -> ProcessingFunctionType:
        return ProcessingFunctionType.HR

