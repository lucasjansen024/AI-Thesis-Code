import datetime
import os
from pathlib import Path
from typing import List, Dict, TypeVar
from pathlib import Path

import numpy as np
import pandas as pd
import re
from .data_types.visit_config import Config
from .preparation.config import config_lucas
from .constants import SEQUENCE_TIME, SEQUENCEV2_TIME, SEQUENCE_UNITS, PRBS_UNIT_LENGTH, PRBS_WARMUP_TIME, VISITS_ALL, PARTICIPANTS_ALL
from .data_types.base import File, Participant, Visit, TrainingData

class Utils:
    @staticmethod
    def prbs(input='101111'):
        output = []
        for i in range(SEQUENCE_UNITS):
            sum = int(input[0]) + int(input[len(input) - 1])
            val = sum % 2
            output.append(val)
            input = str(val) + input[0:len(input) - 1]

        return output

    @staticmethod
    def prbsv2(
            input='10000',
            length=31,
            taps=None
    ):
        if taps is None:
            taps = [4, 1]

        input = [int(x) for x in input]
        output = []

        for _ in range(length):
            output.append(input[-1])
            # Feedback bit (XOR of tap bits)
            feedback = 0
            for t in taps:
                feedback ^= input[t]
            input = [feedback] + input[:-1]

        return output

    @staticmethod
    def to_sequence(units, lower, upper):
        sequence = [lower if i == 0 else upper for i in units]
        return np.repeat(sequence, PRBS_UNIT_LENGTH)

    @staticmethod
    def split_path(path: Path):
        return path.parts

    @staticmethod
    def get_visit_from_visit_path(visit_folder: Path) -> Visit:
        return Visit(Utils.split_path(visit_folder)[-1])

    @staticmethod
    def get_participant_from_visit_path(visit_folder: Path) -> Participant:
        return Participant(Utils.split_path(visit_folder)[-2])

    @staticmethod
    def get_participant_from_participant_folder_path(participant_folder_path: Path) -> Participant:
        split = Utils.split_path(participant_folder_path)
        return Participant(split[-1])

    @staticmethod
    def get_visit_details(visit_path: Path) -> tuple[Participant, Visit]:
        participant = Utils.get_participant_from_visit_path(visit_path)
        visit = Utils.get_visit_from_visit_path(visit_path)
        return participant, visit

    @staticmethod
    def order_participants(participants: List[Participant]) -> List[Participant]:
        ordered = []
        for p in PARTICIPANTS_ALL:
            if p in participants:
                ordered.append(p)
        # keep any participants not in the canonical list (legacy compatibility)
        for p in participants:
            if p not in ordered:
                ordered.append(p)
        return ordered

    @staticmethod
    def order_visits(visits: List[Visit]) -> List[Visit]:
        ordered = []
        for v in VISITS_ALL:
            if v in visits:
                ordered.append(v)
        for v in visits:
            if v not in ordered:
                ordered.append(v)
        return ordered

    @staticmethod
    def check_file_exists(visit_path: Path, file: File) -> bool:
        if file == File.Oxy or file == File.Occlusion:
            participant, visit = Utils.get_visit_details(visit_path)
            if config_lucas.get(participant.value, {}).get(visit.value, {}).get('NIRS_repetition') is None:
                return False

        for visit_file in os.listdir(visit_path):
            if file in visit_file:
                return True

        return False

    @staticmethod
    def get_csv_from_File(visit_path: Path, file: File) -> pd.DataFrame | None:
        for visit_file in os.listdir(visit_path):
            if file in visit_file:
                return pd.read_csv(visit_path / visit_file)
        return None

    @staticmethod
    def remove_prefix(text, prefix) -> str:
        return text[text.startswith(prefix) and len(prefix):]

    @staticmethod
    def interpolate_dataframe(df, column='t'):
        full_range = pd.Series(np.arange(0, df[column].max() + 1, 1), name=column)
        df = df.drop_duplicates(subset=column, keep='first')  # Drop duplicates from 't' column
        df = df.set_index(column).reindex(full_range).reset_index()
        df = df.apply(pd.to_numeric, errors='coerce').interpolate(method='linear', limit_direction='backward')
        df.drop(columns=[column], inplace=True)
        return df

    @staticmethod
    def string_in_element_of(string: str, array: List[iter]):
        for element in array:
            if string in element:
                return True
        return False

    @staticmethod
    def element_of_in_string(array: List[any], string: str):
        for element in array:
            if element in string:
                return True
        return False

    @staticmethod
    def is_participant(string: str):
        return re.match(r'P\d+', string) is not None

    T = TypeVar("T")

    @staticmethod
    def get_first_visit_data(data: Dict[Participant, Dict[Visit, T]]) -> T:
        first_participant_dict = next(iter(data.values()))
        return next(iter(first_participant_dict.values()))

    @staticmethod
    def train_test_split_LOOCV(
            training_data: TrainingData,
            test_participant: Participant,
            train_visits: List[Visit] | None = None,
            test_visits: List[Visit] | None = None,
    ):
        X_train, y_train, X_test, y_test = [], [], [], []
        for participant in training_data:
            for visit in training_data[participant]:
                visit_data = training_data[participant][visit]

                if participant == test_participant:
                    if test_visits is not None and visit not in test_visits:
                        continue
                    X_test.append(visit_data.X)
                    y_test.append(visit_data.y)
                else:
                    if train_visits is not None and visit not in train_visits:
                        continue
                    X_train.append(visit_data.X)
                    y_train.append(visit_data.y)

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError(f"Empty split for test participant {test_participant}. Train visits={train_visits}, test visits={test_visits}")

        X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
        X_test, y_test = np.concatenate(X_test), np.concatenate(y_test)
        return X_train, y_train, X_test, y_test

    @staticmethod
    def remove_warmup(
            df: pd.DataFrame,
            warmup_length=PRBS_WARMUP_TIME
    ) -> pd.DataFrame:
        res = df.iloc[warmup_length:]
        res.reset_index(drop=True, inplace=True)
        return res

    # Average both repetitions of the same visit,
    # assumes warmup is already removed
    @staticmethod
    def average_repetitions(
            df: pd.DataFrame,
            repetition_length=SEQUENCEV2_TIME
    ) -> pd.DataFrame:
        length = len(df.index)
        res = None
        if length == repetition_length:
            res = df.copy()
        elif length < repetition_length * 2:
            diff = repetition_length * 2 - length
            averaged_incomplete = Utils.average_dataframes(df.iloc[:repetition_length - diff], df.iloc[repetition_length:])
            remainder = df.iloc[repetition_length - diff:repetition_length]
            res = pd.concat([averaged_incomplete, remainder])
        else:
            res = Utils.average_dataframes(df.iloc[:repetition_length], df.iloc[repetition_length: repetition_length * 2])

        res.reset_index(drop=True, inplace=True)
        return res

    @staticmethod
    def get_half_repetition(
            df: pd.DataFrame,
            first_half=True,
            repetition_length=SEQUENCEV2_TIME
    ) -> pd.DataFrame:
        length = len(df.index)
        res = None
        if length == repetition_length:
            res = df.copy()
        elif length != repetition_length * 2:
            diff = repetition_length * 2 - length
            if first_half:
                res = df.iloc[:repetition_length]
            else:
                remainder = df.iloc[repetition_length - diff:repetition_length]
                res = pd.concat([df.iloc[repetition_length - diff:], remainder])
        else:
            if first_half:
                res = df.iloc[:repetition_length]
            else:
                res = df.iloc[repetition_length:]

        res.reset_index(drop=True, inplace=True)
        return res

    @staticmethod
    def average_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        return (df1.reset_index() + df2.reset_index()) / 2

    @staticmethod

    def get_file(path, keyword):
        path = Path(path).resolve()

        for file in path.iterdir():
            if keyword in file.name:
                return file

    @staticmethod
    def VO2_time_to_sec(t):
        return float((t.hour * 60 + t.minute) * 60 + t.second + t.microsecond / 1_000_000)

    @staticmethod
    def VO2_string_to_sec(t: str):
        t = re.sub(r'(\d+):(\d{2}):(\d{2})[,.](\d+)', r'\1:\2:\3.\4', t)
        t = datetime.datetime.strptime(t, "%H:%M:%S.%f")
        return Utils.VO2_time_to_sec(t)

    @staticmethod
    def calculate_VO2Max(df: pd.DataFrame, time=30, VO2_col='VO2', time_col='t'):
        df = df.sort_values(by=time_col).reset_index(drop=True)

        max_mean_vo2 = 0
        start_idx = 0

        for end_idx in range(len(df)):
            # Slide the start index forward to maintain window within `time` seconds
            while df.loc[end_idx, time_col] - df.loc[start_idx, time_col] >= time:
                start_idx += 1

            window_vo2 = df.loc[start_idx:end_idx, VO2_col]
            mean_vo2 = window_vo2.mean()
            max_mean_vo2 = max(max_mean_vo2, mean_vo2)

        return max_mean_vo2


