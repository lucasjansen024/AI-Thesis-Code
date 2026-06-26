import copy
import os
import re
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Dict, List, TypeVar

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src2_training_ready.data_types.visit_config import NIRSRepetition
from ..constants import *
from ..data_types.file_keys import FitFileKeys
from ..preparation.config import config_lucas

from ..processing.processors import VO2
from ..processing.processor import Processor
from ..data_types.base import Participant, File, PreparedVisitData, \
    PreparedData, ProcessedData, ProcessedVisitData, Visit, TrainingData, TrainingVisitData, \
    ParticipantDetail, ParticipantDetails
from ..models.models import TCN_30
from ..models.model import Model
from ..utils import Utils


class DataManager:

    def __init__(
            self,
            required_files: List[File],
            visit_selection: VisitSelection = VisitSelection.ALL_VISITS,
            model: Model = TCN_30,
            participant_selection: ParticipantSelection = ParticipantSelection.ALL_PARTICIPANTS,
            logging=True,
    ):
        self.required_files = required_files

        self.visit_selection = visit_selection
        self.participant_selection = participant_selection

        self.visits = visit_selection.value
        self.target_variable = "VO2"
        self.participants = Utils.order_participants(participant_selection.value)
        self.model: Model = model
        self.__logging = logging

        self.prepared_data: PreparedData = {}
        self.prepared = False

        self.processed_data: ProcessedData = {}
        self.processed = False

        self.scalars: Dict[str, MinMaxScaler | StandardScaler] = {}
        self.non_scaling_variables = ['gender']
        self.min_max_scaling_variables = ['WR']
        self.training_data: TrainingData = {}
        self.prepared_training_data = False

    def _log(self, message: str):
        if self.__logging:
            print(f"{datetime.now().strftime('%H:%M:%S:%f')} DataManager Info: {message}")

    def _warn(self, message: str):
        print(f"{datetime.now().strftime('%H:%M:%S:%f')} DataManager Warning: {message}")

    @staticmethod
    def load_participant_details(participant_selection=ParticipantSelection.ALL_PARTICIPANTS) -> ParticipantDetails:
        participant_details: ParticipantDetails = {}
        for participant in participant_selection.value:
            participant = Participant(participant)
            RAMP_path = RAW_PATH / participant / Visit.Ramp
            MRT_file = pd.read_excel(Utils.get_file(RAMP_path, 'MRT'))

            RAMP_file = pd.read_excel(Utils.get_file(RAMP_path, 'BxB'))

            VO2Max = MRT_file.iloc[19, 10]
            VO2GET = MRT_file.iloc[19, 9]
            WRMax = MRT_file.iloc[19, 11]
            WRGET = MRT_file.iloc[20, 4]
            MRT = MRT_file.iloc[15, 4]
            m = MRT_file.iloc[28, 6]
            c = MRT_file.iloc[28, 7]
            ramp_rate = int(re.search(r'\d+(?=(\s)?[wW])', RAMP_file.iloc[11, 4]).group(0))

            weights = []
            height = 0
            age = 0
            for visit in os.listdir(PREPARED_PATH / participant):
                visit = Visit(visit)
                visit_path = PREPARED_PATH / participant / visit

                Fit_path = Utils.get_file(visit_path, File.Fit)
                Fit_file = pd.read_csv(Fit_path)
                weights.append(Fit_file[FitFileKeys.weight].values[0])
                height = Fit_file[FitFileKeys.height].values[0]
                age = Fit_file[FitFileKeys.age].values[0]

            details = ParticipantDetail(
                name=participant,
                weight=np.mean(weights),
                height=height,
                age=age,
                m=m,
                c=c,
                ramp_rate=ramp_rate,
                VO2Max=VO2Max,
                VO2GET=VO2GET,
                WRMax=WRMax,
                WRGET=WRGET,
                MRT=MRT,
            )
            participant_details[participant] = details

        return participant_details

    def load_prepared_data(self):
        self.prepared_data = {}
        self._log("Loading prepared data")
        for participant in self.participants:
            participant = Participant(participant)
            self.prepared_data[participant] = {}
            for visit in os.listdir(PREPARED_PATH / participant):
                visit = Visit(visit)
                if visit not in self.visits:
                    continue

                visit_path = PREPARED_PATH / participant / visit
                if not self.__check_visit_files_exist(visit_path):
                    continue

                visit_files = self.__get_visit_files(visit_path)
                self.prepared_data[participant][visit] = PreparedVisitData(participant, visit, visit_files)

        self.__fill_in_missing_data()
        self.prepared = True

    def __check_visit_files_exist(self, visit_path: Path) -> bool:
        for file in self.required_files:
            if not Utils.check_file_exists(visit_path, file):
                return False
        return True

    def __get_visit_files(self, visit_path: Path) -> Dict[File, pd.DataFrame]:
        csvs = [Utils.get_csv_from_File(visit_path, file) for file in self.required_files]
        return dict(zip(self.required_files, csvs))

    def __fill_in_missing_data(self):
        return

    def process_data(self, processor: Processor):
        if not self.prepared:
            self._warn(f"Prepared data not loaded, loading now")
            self.load_prepared_data()

        self._log("Processing data")
        for participant in self.prepared_data:
            self.processed_data[participant] = {}
            for visit in self.prepared_data[participant]:
                prepared_visit_data = self.prepared_data[participant][visit]
                processed_visit_data = self.__process_prepared_visit(processor, prepared_visit_data)
                self.processed_data[participant][visit] = processed_visit_data
        self.processed = True

    def __process_prepared_visit(
            self,
            processor: Processor,
            prepared_visit_data: PreparedVisitData,
    ) -> ProcessedVisitData:
        visit_data = processor.apply(prepared_visit_data)
        visit_data.data = Utils.interpolate_dataframe(visit_data.data)
        visit_data = self.__trim_processed_visit(visit_data)

        if visit_data.data.isnull().values.any():
            raise ValueError(f"{visit_data.data.isnull().sum()} NaN values in processed data")

        return visit_data

    def __trim_processed_visit(
            self,
            visit_data: ProcessedVisitData
    ) -> ProcessedVisitData:
        participant = visit_data.participant
        visit = visit_data.visit
        data = visit_data.data

        participant_key = participant.value if hasattr(participant, 'value') else str(participant)
        visit_key = visit.value if hasattr(visit, 'value') else str(visit)
        visit_config = config_lucas.get(participant_key, {}).get(visit_key, {})

        # Keep the same warmup removal behavior (with receptive field context), but use Lucas submax total duration.
        if visit.is_ramp():
            start = max((REST_TIME + RAMP_WARMUP_TIME) - self.model.receptive_field, 0)
        else:
            start = max((REST_TIME + PRBS_WARMUP_TIME) - self.model.receptive_field, 0)
        end = SUBMAX_TOTAL_TIME if not visit.is_ramp() else None

        requires_NIRS = File.Oxy in self.required_files or File.Occlusion in self.required_files
        nirs_rep = visit_config.get('NIRS_repetition')
        if requires_NIRS and nirs_rep == NIRSRepetition.One.value and end is not None:
            # Lucas submax protocols are 2 x 15:30 (930 s) repetitions.
            end -= SEQUENCEV2_TIME

        if end is None:
            data = data[start:]
        else:
            data = data[start:end]

        data = data.dropna()
        data.reset_index(drop=True, inplace=True)
        visit_data.data = data
        return visit_data

    def prepare_training_data(self):
        if not self.processed:
            raise ValueError("Data not processed")

        self._log("Preparing training data")
        self.__create_scalars()
        normalized_data = self.__normalize_processed_data()
        self.__split_normalized_data(normalized_data)
        self.prepared_training_data = True

    def __create_scalars(self):
        for participant in self.processed_data:
            for visit in self.processed_data[participant]:
                visit_data = self.processed_data[participant][visit]
                for col in visit_data.data.columns:
                    if Utils.element_of_in_string(self.non_scaling_variables, col):
                        continue
                    if col not in self.scalars.keys():
                        if Utils.element_of_in_string(self.min_max_scaling_variables, col):
                            self.scalars[col] = MinMaxScaler()
                        else:
                            self.scalars[col] = StandardScaler()
                    self.scalars[col].partial_fit(visit_data.data[col].values.reshape(-1, 1))

    def __normalize_processed_data(self) -> ProcessedData:
        normalized_data: ProcessedData = {}
        for participant in self.processed_data:
            normalized_data[participant] = {}
            for visit in self.processed_data[participant]:
                visit_data = self.processed_data[participant][visit]
                copy = deepcopy(visit_data)
                for col in visit_data.data.columns:
                    if Utils.element_of_in_string(self.non_scaling_variables, col):
                        copy.data[col] = visit_data.data[col]
                    else:
                        copy.data[col] = self.scalars[col].transform(copy.data[col].values.reshape(-1, 1))
                normalized_data[participant][visit] = copy
        return normalized_data

    def __split_normalized_data(self, normalized_data: ProcessedData):
        for participant in normalized_data:
            self.training_data[participant] = {}
            for visit in normalized_data[participant]:
                visit_data = normalized_data[participant][visit]
                X, y = [], []

                for col in visit_data.data.columns:
                    if 'VO2' in col:
                        y = visit_data.data[col].values
                    else:
                        X.append(visit_data.data[col].values)

                X = np.array(X).T
                X, y = self.__split_visit_data(X, y)
                self.training_data[participant][visit] = TrainingVisitData(participant, visit, X, y)

    def __split_visit_data(self, X, y):
        X_split = []
        y_split = []
        for i in range(self.model.receptive_field, len(X)):
            X_split.append(X[i - self.model.receptive_field:i])
            y_split.append(y[i])
        return np.array(X_split), np.array(y_split)

    def _subset_processed_data(self, participants: list[Participant] | None = None, visits: list[Visit] | None = None) -> ProcessedData:
        if not self.processed:
            raise ValueError("Data not processed")

        participants = participants if participants is not None else list(self.processed_data.keys())
        subset: ProcessedData = {}
        for participant in participants:
            if participant not in self.processed_data:
                continue
            subset[participant] = {}
            for visit, visit_data in self.processed_data[participant].items():
                if visits is not None and visit not in visits:
                    continue
                subset[participant][visit] = copy.deepcopy(visit_data)
        return subset

    def _fit_scalars_on_processed_data(self, processed_subset: ProcessedData):
        scalers: Dict[str, MinMaxScaler | StandardScaler] = {}
        for participant in processed_subset:
            for visit in processed_subset[participant]:
                visit_df = processed_subset[participant][visit].data
                for col in visit_df.columns:
                    if Utils.element_of_in_string(self.non_scaling_variables, col):
                        continue
                    if col not in scalers:
                        if Utils.element_of_in_string(self.min_max_scaling_variables, col):
                            scalers[col] = MinMaxScaler()
                        else:
                            scalers[col] = StandardScaler()
                    scalers[col].partial_fit(visit_df[[col]])
        return scalers

    def _apply_scalars_to_processed_data(self, processed_subset: ProcessedData, scalers: Dict[str, MinMaxScaler | StandardScaler]) -> ProcessedData:
        normalized_data: ProcessedData = copy.deepcopy(processed_subset)
        for participant in normalized_data:
            for visit in normalized_data[participant]:
                visit_data = normalized_data[participant][visit]
                for col in visit_data.data.columns:
                    if col in scalers:
                        visit_data.data[col] = scalers[col].transform(visit_data.data[[col]])
        return normalized_data

    def _processed_to_training_data(self, processed_subset: ProcessedData) -> TrainingData:
        training_data: TrainingData = {}
        for participant in processed_subset:
            training_data[participant] = {}
            for visit in processed_subset[participant]:
                visit_data = processed_subset[participant][visit]
                X = visit_data.data.drop(columns=[self.target_variable]).to_numpy()
                y = visit_data.data[[self.target_variable]].to_numpy()
                X_split, y_split = self.__split_visit_data(X, y)
                training_data[participant][visit] = TrainingVisitData(
                    participant=participant,
                    visit=visit,
                    X=X_split,
                    y=y_split,
                )
        return training_data

    
    def _debug_feature_shift(self, test_participant, train_visits, test_visits):
        train_participants = [p for p in self.processed_data.keys() if p != test_participant]

        train_processed = self._subset_processed_data(participants=train_participants, visits=train_visits)
        test_processed = self._subset_processed_data(participants=[test_participant], visits=test_visits)

        train_processed = {p: v for p, v in train_processed.items() if len(v) > 0}
        test_processed = {p: v for p, v in test_processed.items() if len(v) > 0}

        scalers = self._fit_scalars_on_processed_data(train_processed)

        def collect_col(proc_dict, col):
            vals = []
            for p in proc_dict:
                for v in proc_dict[p]:
                    if col in proc_dict[p][v].data.columns:
                        vals.append(proc_dict[p][v].data[col].to_numpy())
            return np.concatenate(vals)

        example_participant = next(iter(test_processed))
        example_visit = next(iter(test_processed[example_participant]))
        cols = list(test_processed[example_participant][example_visit].data.columns)

        #print(f"\n=== FEATURE SHIFT DEBUG for {test_participant} ===")
        for col in cols:
            train_vals = collect_col(train_processed, col)
            test_vals = collect_col(test_processed, col)

            if col in scalers:
                scaler = scalers[col]
                test_scaled = scaler.transform(test_vals.reshape(-1, 1)).reshape(-1)
                #print(
                #    f"{col:12s} | "
                #    f"train raw [{train_vals.min():8.3f}, {train_vals.max():8.3f}] "
                #    f"mean={train_vals.mean():8.3f} | "
                #    f"test raw [{test_vals.min():8.3f}, {test_vals.max():8.3f}] "
                #    f"mean={test_vals.mean():8.3f} | "
                #    f"test scaled [{test_scaled.min():8.3f}, {test_scaled.max():8.3f}] "
                #    f"mean={test_scaled.mean():8.3f}"
                #)
            #else:
                #print(
                #    f"{col:12s} | "
                #    f"train raw [{train_vals.min():8.3f}, {train_vals.max():8.3f}] "
                #    f"mean={train_vals.mean():8.3f} | "
                #    f"test raw [{test_vals.min():8.3f}, {test_vals.max():8.3f}] "
                #    f"mean={test_vals.mean():8.3f}"
                #)
    def get_fold_data(
            self,
            test_participant: Participant,
            train_visits: list[Visit] | None = None,
            test_visits: list[Visit] | None = None,
    ):
        """Leakage-free fold creation: fit scalers on train participants only, then transform train+test."""
        if not self.processed:
            raise ValueError("Data not processed")

        all_participants = list(self.processed_data.keys())
        train_participants = [p for p in all_participants if p != test_participant]
        if len(train_participants) == 0:
            raise ValueError("Need at least 2 participants for LOOCV")

        train_processed = self._subset_processed_data(participants=train_participants, visits=train_visits)
        test_processed = self._subset_processed_data(participants=[test_participant], visits=test_visits)

        # Remove empty participants (e.g., when a protocol is missing)
        train_processed = {p: v for p, v in train_processed.items() if len(v) > 0}
        test_processed = {p: v for p, v in test_processed.items() if len(v) > 0}
        if len(train_processed) == 0 or len(test_processed) == 0:
            raise ValueError(
                f"Empty fold split for {test_participant}. train_visits={train_visits}, test_visits={test_visits}"
            )
        
        #print(f"\n=== Fold test participant: {test_participant} ===")
        for p in train_processed:
            for v in train_processed[p]:
                vo2 = train_processed[p][v].data['VO2']
                #print(f"TRAIN {p} {v}: VO2 min={vo2.min():.2f}, max={vo2.max():.2f}, mean={vo2.mean():.2f}")

        for p in test_processed:
            for v in test_processed[p]:
                vo2 = test_processed[p][v].data['VO2']
                #print(f"TEST  {p} {v}: VO2 min={vo2.min():.2f}, max={vo2.max():.2f}, mean={vo2.mean():.2f}")


        self._debug_feature_shift(test_participant, train_visits, test_visits)
        scalers = self._fit_scalars_on_processed_data(train_processed)
        train_norm = self._apply_scalars_to_processed_data(train_processed, scalers)
        test_norm = self._apply_scalars_to_processed_data(test_processed, scalers)

        train_td = self._processed_to_training_data(train_norm)
        test_td = self._processed_to_training_data(test_norm)

        X_train, y_train, X_test, y_test = [], [], [], []
        for p in train_td:
            for v in train_td[p]:
                X_train.append(train_td[p][v].X)
                y_train.append(train_td[p][v].y)
        for p in test_td:
            for v in test_td[p]:
                X_test.append(test_td[p][v].X)
                y_test.append(test_td[p][v].y)

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError(
                f"No windows generated for fold {test_participant}. train_visits={train_visits}, test_visits={test_visits}"
            )

        return (
            np.concatenate(X_train),
            np.concatenate(y_train),
            np.concatenate(X_test),
            np.concatenate(y_test),
            scalers,
        )
    @property
    def processed_data_with_raw_VO2(self):
        data_manager = DataManager(
            required_files=[File.VO2BxB],
            visit_selection=self.visit_selection,
            participant_selection=self.participant_selection,
            model=self.model,
            logging=False,
        )
        data_manager.process_data(VO2)

        data = copy.deepcopy(self.processed_data)
        for participant in data:
            for visit in data[participant]:
                visit_data = data[participant][visit]
                visit_data.data['raw'] = data_manager.processed_data[participant][visit].data['VO2']

        return data


