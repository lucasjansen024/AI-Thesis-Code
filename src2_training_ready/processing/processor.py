import inspect
from typing import List, OrderedDict

import pandas as pd
import numpy as np

from .function import ProcessingFunction
from ..utils import Utils
from ..constants import TOTAL_TIME
from ..data_types import PreparedVisitData, File, ProcessedVisitData


class Processor:

    def __init__(self):
        self.processing_functions: List[ProcessingFunction] = []

    @property
    def sorted_functions(self) -> List[ProcessingFunction]:
        return sorted(self.processing_functions, key=lambda pf: pf.type.value)

    def add(self, processing_function: ProcessingFunction):
        if not isinstance(processing_function, ProcessingFunction): # If it is not instantiated yet
            processing_function = processing_function()
        self.processing_functions.append(processing_function)
        return self

    def has(self, processing_function: ProcessingFunction):
        for func in self.processing_functions:
            if func == processing_function:
                return True
        return False

    def remove(self, processing_function: ProcessingFunction):
        for func in self.processing_functions:
            if func == processing_function:
                self.processing_functions.remove(func)
        return False

    def apply(self, visit_data: PreparedVisitData) -> ProcessedVisitData:
        res = pd.DataFrame({'t': np.arange(0, TOTAL_TIME * 2)})
        for processing_function in self.sorted_functions:
            df = processing_function.apply(visit_data)

            # Ensure df has a 't' column
            if 't' not in df.columns:
                df['t'] = df.index

            # Merge df into res DataFrame
            res = res.merge(df, on='t', how='left')

        # Find the last index where any column other than 't' is not NaN
        last_valid_index = res.loc[:, res.columns != 't'].dropna(how='all').index[-1]
        res = res.loc[:last_valid_index]
        return ProcessedVisitData(visit_data.participant, visit_data.visit, res)

    def get_required_files(self) -> List[File]:
        required_files = []
        for processing_function in self.sorted_functions:
            required_files.extend(processing_function.required_files())

        # Deduplicate while preserving order
        seen = OrderedDict()
        for f in required_files:
            seen[f] = None

        return list(seen.keys())

    def get_folder_name(self):
        names = [Utils.remove_prefix(processing_function.__class__.__name__, 'Process_') for processing_function in self.processing_functions]
        return '-'.join(sorted(names))


