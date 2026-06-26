import copy
import dataclasses
import json
import pickle
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List, Dict, Optional

import dacite
from dacite import from_dict

import numpy as np
import pandas as pd
import shap
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import keras

from ..data_types.base import ProcessedData, TrainingData, Participant, ProcessedVisitData, ResultsData, Visit
from ..data_types.results import ShapResult, ParticipantModelMetric, \
    SetOfModelsResult, ParticipantModelResult, BlandAltmanResult, PearsonResult, RMCorrResult, ResidualRunsTestCollection, ResidualRunsTestResult
from ..models.TCN import TCN, ResidualBlock
from ..analysis import Analysis
from ..constants import COMMON_FOLDER, BEST_FOLDER, RESULTS_FOLDER
from ..models.model import Model
from ..utils import Utils
from .data_manager import DataManager

@dataclasses.dataclass
class CurrentPaths:
    model_folder_path: Path
    participant_folder_paths: List[Path]
    results_path: Path

@dataclasses.dataclass
class Dataframes:
    VO2_df: pd.DataFrame
    MNG_df: pd.DataFrame
    boxplot_df: pd.DataFrame
    MNG_df_averaged: pd.DataFrame
    boxplot_df_averaged: pd.DataFrame


class ResultsManager:

    def __init__(
            self,
            base_path: Path,
            model: Model,
            cut_off_frequency=0.01,
            logging=True,
            train_visits: List[Visit] | None = None,
            test_visits: List[Visit] | None = None,
    ):
        self.__base_path = base_path

        # Data provided from outside using initialize function
        self.__original_processed_data: ProcessedData | None = None
        self.__training_data: TrainingData | None = None
        self.__scalars: Dict[str, MinMaxScaler | StandardScaler] | None = None
        self.__data_manager: DataManager | None = None
        self.__initialized = False

        # Data generated when a model is selected for testing
        self.processed_data: ResultsData | None = None
        self.averaged_processed_data: ResultsData | None = None
        self.__current_paths: CurrentPaths | None = None
        self.__current_dataframes: Dataframes | None = None

        self.__model: Model = model
        self.raw_VO2_column = 'raw'
        self.WR_column = 'WR'
        self.VO2_column = 'VO2'
        self.predictions_column = 'predictions'

        self.__cut_off_frequency = cut_off_frequency
        self.__train_visits = train_visits
        self.__test_visits = test_visits

        self.__results_file = 'results.json'
        self.__aggregated_results_file = 'aggregated_results.csv'
        self.__participant_model_results_file = 'participant_model_results.json'
        self.__residual_runs_test_file = 'VO2_residual_runs_test.csv'
        # self.__test_losses_file = 'test_losses.json'
        self.__ignore_folders = [BEST_FOLDER, COMMON_FOLDER, 'results']
        self.__logging = logging

    def _log(self, message: object):
        if self.__logging:
            print(f"{datetime.now().strftime('%H:%M:%S:%f')} Manager Info: {message}")

    def _warn(self, message: str):
        print(f"{datetime.now().strftime('%H:%M:%S')} ResultsManager Warning: {message}")

    def set_data(
            self,
            processed_data: ProcessedData,
            training_data: TrainingData | None,
            scalars: Dict[str, MinMaxScaler | StandardScaler] | None,
            data_manager: DataManager | None = None,
    ):
        self.__original_processed_data = processed_data
        self.__training_data = training_data
        self.__scalars = scalars
        self.__data_manager = data_manager
        self.__initialized = True
        self._invalidate_calculated_data()

    def _invalidate_calculated_data(self):
        self.processed_data = None
        self.averaged_processed_data = None
        self.__current_paths = None
        self.__current_dataframes = None

    def _get_fold_data(self, participant: Participant):
        if self.__data_manager is not None:
            return self.__data_manager.get_fold_data(
                participant,
                train_visits=self.__train_visits,
                test_visits=self.__test_visits,
            )

        if self.__training_data is None:
            raise ValueError("ResultsManager has no training data or DataManager")

        X_train, y_train, X_test, y_test = Utils.train_test_split_LOOCV(
            self.__training_data,
            participant,
            train_visits=self.__train_visits,
            test_visits=self.__test_visits,
        )
        return X_train, y_train, X_test, y_test, self.__scalars
    
    

    def _load_fold_scalers(self, participant_folder: Path):
        fold_scalers_path = participant_folder / 'fold_scalars.pkl'
        if fold_scalers_path.exists():
            return pickle.load(open(fold_scalers_path, 'rb'))
        return self.__scalars

    def _inverse_transform_vo2(self, values: np.ndarray, scalers: Dict[str, MinMaxScaler | StandardScaler] | None):
        if scalers is None or self.VO2_column not in scalers:
            return values.reshape(-1)
        scaler = scalers[self.VO2_column]
        arr = np.asarray(values).reshape(-1, 1)
        return scaler.inverse_transform(arr).reshape(-1)

    def get_amount_of_trained_models(
            self,
            exclude_participants: List[Participant] = []
    ):
        total = 0
        all_models = self.get_all_model_folders()
        for model_folder in all_models:
            results_json_path = self.get_results_path(model_folder, exclude_participants) / self.__results_file
            if not os.path.exists(results_json_path):
                continue
            total += 1
        return total

    def get_processed_data(self) -> ResultsData:
        return self.processed_data

    def get_all_model_folders(self) -> List[str]:
        models = []
        if os.path.exists(self.__base_path):
            for model_folder in os.listdir(self.__base_path):
                if Utils.string_in_element_of(model_folder, self.__ignore_folders):
                    continue
                models.append(model_folder)
        return models

    def test_all(
            self,
            exclude_participants: List[Participant] = []
    ) -> Dict[str, SetOfModelsResult]:
        self._log(f"Testing all models in {self.__base_path}, excluding {exclude_participants}")
        results = {}
        for model_folder in self.get_all_model_folders():
            results[model_folder] = (self.test_folder(model_folder, exclude_participants))
        return results

    def load_all(
            self,
            include_best=True,
            exclude_participants: List[Participant] = []
    ) -> Dict[str, SetOfModelsResult | None]:
        self._log(f"Loading all results in {self.__base_path}, excluding {exclude_participants}")
        results = {}
        for model_folder in self.get_all_model_folders():
            results[model_folder] = self.load_results(model_folder, exclude_participants)

        if include_best:
            results[BEST_FOLDER] = self.load_results(BEST_FOLDER)

        return results

    def mean_results(
            self,
            exclude_participants: List[Participant] = []
    ) -> SetOfModelsResult | None:
        self._log(f"Loading mean results in {self.__base_path}, including {exclude_participants}")
        results = self.load_all(include_best=False, exclude_participants=exclude_participants)

        if results is None:
            return None

        mean_results = SetOfModelsResult(
            VO2_bland_altman_global=BlandAltmanResult(bias=0, sd=0),
            VO2_pearson=PearsonResult(r=0, p=0),
            VO2_rm_corr=RMCorrResult(r=0, p=0),
            MNG_rm_corr=RMCorrResult(r=0, p=0),
            MNG_bland_altman_global=BlandAltmanResult(bias=0, sd=0),
            MNG_rm_corr_averaged=RMCorrResult(r=0, p=0),
            MNG_bland_altman_global_averaged=BlandAltmanResult(bias=0, sd=0),
            VO2_residual_runs_test=None
        )

        count = len(results)
        for model_folder in results:
            if results[model_folder] is None:
                continue
            mean_results.VO2_bland_altman_global.bias += results[model_folder].VO2_bland_altman_global.bias / count
            mean_results.VO2_bland_altman_global.sd += results[model_folder].VO2_bland_altman_global.sd / count
            mean_results.VO2_pearson.r += results[model_folder].VO2_pearson.r / count
            mean_results.VO2_rm_corr.r += results[model_folder].VO2_rm_corr.r / count
            mean_results.MNG_rm_corr.r += results[model_folder].MNG_rm_corr.r / count
            mean_results.MNG_bland_altman_global.bias += results[model_folder].MNG_bland_altman_global.bias / count
            mean_results.MNG_bland_altman_global.sd += results[model_folder].MNG_bland_altman_global.sd / count

        return mean_results

    def test_indexed_folder(
            self,
            index: int,
            exclude_participants: List[Participant] = []
    ) -> SetOfModelsResult:
        model_folder = self.get_model_folder_from_index(index)
        return self.test_folder(model_folder, exclude_participants)

    def load_index_results(
            self,
            index: int,
            exclude_participants: List[Participant] = []
    ) -> SetOfModelsResult:
        model_folder = self.get_model_folder_from_index(index)
        return self.load_results(model_folder, exclude_participants)

    def prepare_indexed_data(self, index: int, exclude_participants: List[Participant] = []):
        model_folder = self.get_model_folder_from_index(index)
        self.prepare_data(model_folder, exclude_participants)

    def get_model_folder_from_index(self, index: int) -> str:
        model_folder = ''
        for i, participant_folder in enumerate(os.listdir(self.__base_path)):
            if i == index:
                model_folder = participant_folder
                break

        if model_folder == '':
            raise IndexError(f"Model folder with index {index} does not exist")

        return model_folder

    def get_model_folder_path(self, model_folder: str) -> Path:
        if model_folder == COMMON_FOLDER:
            raise ValueError(f"Model folder {model_folder} is not a valid model folder")

        model_folder_path = self.__base_path / model_folder

        if model_folder == BEST_FOLDER:
            model_folder_path.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(model_folder_path):
            raise FileNotFoundError(f"Model folder {model_folder} does not exist")

        return model_folder_path

    def test_folder(
            self,
            model_folder: str,
            exclude_participants: List[Participant] = [],
            force=False,
    ) -> SetOfModelsResult:
        self._log(f"Testing model folder {model_folder}, excluding {exclude_participants}")
        is_best = model_folder == BEST_FOLDER
        if not is_best and not force:
            existing_results = self.load_results(model_folder, exclude_participants)
            if existing_results is not None:
                self._log(f"Results for {model_folder} already calculated, returning")
                return existing_results

        model_folder_path = self.get_model_folder_path(model_folder)
        if not self._is_processed_for_model_folder_path(model_folder_path):
            self._prepare_data_for_model_folder_path(model_folder_path, exclude_participants)

        return self._calculate_results_for_current_paths(is_best)

    def load_results(
            self,
            model_folder: str,
            exclude_participants: List[Participant] = []
    ) -> SetOfModelsResult | None:
        results_json_path = self.get_results_path(model_folder, exclude_participants) / self.__results_file
        if not os.path.exists(results_json_path):
            self._warn(f"Results for {model_folder} not calculated yet")
            return None

        json_result = json.load(open(results_json_path, 'r'))
        return from_dict(data_class=SetOfModelsResult, data=json_result, config=dacite.Config(check_types=False, ))

    def prepare_data(self, model_folder: str, exclude_participants: List[Participant] = []):
        model_folder_path = self.get_model_folder_path(model_folder)
        self._log(f"Preparing data for model folder {model_folder}, excluding {exclude_participants}")
        self._prepare_data_for_model_folder_path(model_folder_path, exclude_participants)

    def get_dataframes(
            self,
            model_folder: str,
            exclude_participants: List[Participant] = []
    ) -> Dataframes:
        model_folder_path = self.get_model_folder_path(model_folder)
        if not self._is_processed_for_model_folder_path(model_folder_path):
            self._prepare_data_for_model_folder_path(model_folder_path, exclude_participants)

        return self.__current_dataframes

    def test_best(
            self,
            exclude_participants: List[Participant] = []
    ) -> SetOfModelsResult:
        return self.test_folder(BEST_FOLDER, exclude_participants)

    def load_best(
            self,
            exclude_participants: List[Participant] = []
    ) -> SetOfModelsResult | None:
        return self.load_results(BEST_FOLDER, exclude_participants)

    def load_aggregated_df(
            self,
            exclude_participants: List[Participant] = [],
    ) -> pd.DataFrame | None:
        aggregated_df_path = self.get_results_path(BEST_FOLDER, exclude_participants) / self.__aggregated_results_file
        if not os.path.exists(aggregated_df_path):
            self._warn(f"Aggregated DataFrame for {self.__current_paths.model_folder_path} does not exist")
            return None

        return pd.read_csv(aggregated_df_path)


    def _is_processed_for_model_folder_path(
            self,
            model_folder_path: Path
    ) -> bool:
        if self.__current_paths is None:
            return False
        return self.__current_paths.model_folder_path == model_folder_path

    def _get_participant_folder_paths_from_model_folder_path(
            self,
            model_folder_path: Path,
            exclude_participants: List[Participant]
    ) -> List[Path]:
        if BEST_FOLDER in model_folder_path.parts:
            return self._get_participant_folder_paths_for_best(exclude_participants)

        participant_folder_paths = []
        for participant_folder in os.listdir(model_folder_path):
            if not Utils.is_participant(participant_folder):
                continue
            participant_folder_paths.append(model_folder_path / participant_folder)
        return participant_folder_paths

    def _get_participant_folder_paths_for_best(
            self,
            exclude_participants: List[Participant]
    ) -> List[Path]:
        best_participant_results: Dict[Participant, ParticipantModelMetric] = {}
        for model_folder in os.listdir(self.__base_path):
            if Utils.string_in_element_of(model_folder, self.__ignore_folders):
                continue

            results_path = self.get_results_path(model_folder, exclude_participants)
            individual_model_results_path = results_path / self.__participant_model_results_file
            model_folder_path = self.get_model_folder_path(model_folder)

            # If individual model results not present, calculate them
            if not os.path.exists(individual_model_results_path):
                participant_folders = self._get_participant_folder_paths_from_model_folder_path(model_folder_path,
                                                                                                exclude_participants)
                self._calculate_participant_model_results(participant_folders, results_path)

            # read individual model results and update best_participant_losses
            participant_model_results = json.load(open(individual_model_results_path, "r"))
            for participant in participant_model_results:
                value = participant_model_results[participant]
                participant_model_results[participant] = from_dict(
                    data_class=ParticipantModelResult,
                    data=value,
                    config=dacite.Config(check_types=False)
                )
            self._update_best_participant_results(best_participant_results, model_folder_path, participant_model_results)

        # Select participant_folders from best_participant_losses
        participant_folders = [
            best_participant_results[participant].participant_folder_path
            for participant in best_participant_results
        ]

        if len(participant_folders) == 0:
            raise ValueError(f"No compatible participant folders found for BEST in {self.__base_path}")

        return participant_folders

    def _update_best_participant_results(
            self,
            best_participant_metrics: Dict[Participant, ParticipantModelMetric],
            model_folder_path: Path,
            participant_model_results: Dict[Participant, ParticipantModelResult]
    ):
        for participant in participant_model_results:
            participant_folder_path = model_folder_path / participant
            current_best = best_participant_metrics.get(participant)
            model_result = participant_model_results[participant].rmse
            if (participant not in best_participant_metrics
                    or model_result < current_best.metric):
                best_participant_metrics[participant] = ParticipantModelMetric(model_result, participant_folder_path)

    def get_results_path(
            self,
            model_folder: str,
            exclude_participants: List[Participant] = [],
    ) -> Path:
        results_path = self.get_model_folder_path(model_folder) / RESULTS_FOLDER

        if len(exclude_participants) > 0:
            results_path = results_path / f"exclude_{'_'.join(exclude_participants)}/"

        return results_path

    def _calculate_participant_model_results(
            self,
            participant_folder_paths: List[Path],
            results_path: Path,
    ):
        participant_model_results_path = results_path / self.__participant_model_results_file
        if os.path.exists(participant_model_results_path):
            self._log(f"Participant model results already calculated for {self.__current_paths.model_folder_path}")
            return

        if self.__current_paths is not None:  # If test losses are being calculated when producing results for best folder
            self._log(f"Calculating Participant model results for {self.__current_paths.model_folder_path}")
        else:
            self._log(f"Calculating Participant model results for best")

        individual_model_results = {}
        for p_folder in participant_folder_paths:
            participant = Utils.get_participant_from_participant_folder_path(p_folder)

            model = self._load_model(p_folder)
            X_train, y_train, X_test, y_test, fold_scalers = self._get_fold_data(participant)
            y_pred = model.predict(X_test)
            vo2_scaler = fold_scalers.get('VO2')
            #if vo2_scaler is not None:
                #print(f"\nParticipant {participant}")
                #print("VO2 scaler mean_:", getattr(vo2_scaler, "mean_", None))
                #print("VO2 scaler scale_:", getattr(vo2_scaler, "scale_", None))
                #print("y_test scaled sample:", y_test[:5].reshape(-1))
                #print("y_pred scaled sample:", y_pred[:5].reshape(-1))

            #print("y_pred scaled min/max/mean:", y_pred.min(), y_pred.max(), y_pred.mean())
            #print("y_test scaled min/max/mean:", y_test.min(), y_test.max(), y_test.mean())

            y_test_unscaled = self._inverse_transform_vo2(y_test, fold_scalers)
            y_pred_unscaled = self._inverse_transform_vo2(y_pred, fold_scalers)
            #print("y_test unscaled sample:", y_test_unscaled[:5])
            #print("y_pred unscaled sample:", y_pred_unscaled[:5])
            individual_model_results[participant] = asdict(
                self._calculate_individual_model_result(y_test_unscaled, y_pred_unscaled)
            )

        results_path.mkdir(parents=True, exist_ok=True)
        json.dump(individual_model_results, open(participant_model_results_path, 'w'))

    def _calculate_individual_model_result(self, y_test: ndarray, y_pred: ndarray) -> ParticipantModelResult:
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        pearson = Analysis.pearson(pd.DataFrame({'Measured': y_test, 'Predicted': y_pred}))
        bland_altman = Analysis.bland_altman(y_test, y_pred)
        bland_altman_result = BlandAltmanResult(bland_altman.md, bland_altman.sd)
        return ParticipantModelResult(
            rmse=rmse,
            mse=mse,
            mae=mae,
            pearson=pearson,
            bland_altman=bland_altman_result
        )

    def _prepare_data_for_model_folder_path(
            self,
            model_folder_path: Path,
            exclude_participants: List[Participant],
    ):
        self.__current_paths = None
        participant_folder_paths = self._get_participant_folder_paths_from_model_folder_path(model_folder_path, exclude_participants)
        included_participants = [Utils.get_participant_from_participant_folder_path(p) for p in participant_folder_paths]
        self.processed_data = {
            participant: copy.deepcopy(self.__original_processed_data[participant])
            for participant in included_participants
            if participant in self.__original_processed_data
        }

        # Keep only the test-protocol visits in the results object for cross-protocol evaluation.
        if self.__test_visits is not None:
            for participant in list(self.processed_data.keys()):
                self.processed_data[participant] = {
                    visit: visit_data for visit, visit_data in self.processed_data[participant].items()
                    if visit in self.__test_visits
                }

        for p_folder in participant_folder_paths:
            participant = Utils.get_participant_from_participant_folder_path(p_folder)

            model = self._load_model(p_folder)
            X_train, y_train, X_test, y_test, fold_scalers_runtime = self._get_fold_data(participant)
            #print("=== DEBUG BEST PREDICT ===")
            #print("model_folder_path:", model_folder_path)
            #print("participant:", participant)
            #print("p_folder:", p_folder)
            #print("X_test shape:", X_test.shape)
            #print("loaded model input_shape:", getattr(model, "input_shape", None))
            try:
                predictions = model.predict(X_test, verbose=0)
            except Exception as e:
                raise ValueError(
                    f"Prediction failed for participant {participant} using folder {p_folder}. "
                    f"Model input_shape={getattr(model, 'input_shape', None)}, X_test shape={X_test.shape}. Error: {e}"
                )

            fold_scalers_saved = self._load_fold_scalers(p_folder)
            fold_scalers = fold_scalers_saved or fold_scalers_runtime
            predictions_scaled = self._inverse_transform_vo2(predictions, fold_scalers)
            self._split_predictions_into_visits(predictions_scaled, participant)

        self._generate_averaged_processed_data()
        self.__current_paths = CurrentPaths(
            model_folder_path,
            participant_folder_paths,
            self.get_results_path(model_folder_path.name, exclude_participants)
        )
        self._get_dfs()

    def _calculate_results_for_current_paths(self, best: bool) -> SetOfModelsResult:
        self._calculate_participant_model_results(self.__current_paths.participant_folder_paths, self.__current_paths.results_path)
        results = self._calculate_results()
        self._save_residual_runs_test_csv(results.VO2_residual_runs_test)
        json.dump(asdict(results), open(self.__current_paths.results_path / self.__results_file, 'w'))

        if best:
            aggregated_df = self._get_aggregated_df()
            aggregated_df.to_csv(self.__current_paths.results_path / self.__aggregated_results_file, index=False)

        return results

    def _load_model(self, participant_folder: Path, final_model=False):
        participant = Utils.get_participant_from_participant_folder_path(participant_folder)
        X_train, y_train, X_test, y_test, _ = self._get_fold_data(participant)
    
        input_shape = (X_train.shape[1], X_train.shape[2])
    
        # Voor TCN: nooit load_model gebruiken, alleen model opnieuw maken + weights laden
        if self.__model.name.startswith("TCN"):
            model = self.__model.create_and_compile(input_shape)
    
            best_weights_path = participant_folder / "best_model.weights.h5"
            if best_weights_path.exists():
                model.load_weights(str(best_weights_path))
                return model
    
            raise FileNotFoundError(
                f"No best_model.weights.h5 found in {participant_folder}"
            )
    
        # Voor GRU / andere standaard Keras-modellen kan load_model nog wel
        final_model_path = participant_folder / "final_model.keras"
        best_model_path = participant_folder / "best_model.keras"
    
        if final_model_path.exists():
            return keras.models.load_model(str(final_model_path))
    
        if best_model_path.exists():
            return keras.models.load_model(str(best_model_path))
    
        raise FileNotFoundError(
            f"No model file found in {participant_folder}"
        )

    def _split_predictions_into_visits(self, predictions: ndarray, participant: Participant):
        predictions = np.asarray(predictions).reshape(-1)
        visits = list(self.processed_data[participant].keys())
        visits = Utils.order_visits(visits)
        if self.__test_visits is not None:
            visits = [v for v in visits if v in self.__test_visits]

        for visit in visits:
            processed_visit_data = self.processed_data[participant][visit]
            length = len(processed_visit_data.data.index) - self.__model.receptive_field
            visit_predictions = predictions[:length]
            predictions = predictions[length:]
            processed_visit_data.data = Utils.remove_warmup(processed_visit_data.data, self.__model.receptive_field)
            processed_visit_data.data[self.predictions_column] = visit_predictions

    def _generate_averaged_processed_data(self):
        self.averaged_processed_data = {}

        for participant in self.processed_data:
            self.averaged_processed_data[participant] = {}
            for visit in self.processed_data[participant]:
                visit_data = self.processed_data[participant][visit]
                self.averaged_processed_data[participant][visit] = ProcessedVisitData(
                    participant, visit, Utils.average_repetitions(visit_data.data)
                )

    def _get_dfs(self):
        VO2_df = self._get_VO2_df()
        MNG_df, boxplot_df = self._get_MNG_dfs(self.processed_data)
        MNG_df_averaged, boxplot_df_averaged = self._get_MNG_dfs(self.averaged_processed_data)
        self.__current_dataframes = Dataframes(VO2_df, MNG_df, boxplot_df, MNG_df_averaged, boxplot_df_averaged)

    def _get_aggregated_df(self) -> pd.DataFrame:
        VO2_df = self.__current_dataframes.VO2_df
        VO2_df["sq_error"] = (VO2_df["Predicted"] - VO2_df["Measured"]) ** 2
        VO2_df["abs_error"] = np.abs(VO2_df["Predicted"] - VO2_df["Measured"])

        VO2_df["Participant"] = VO2_df["Participant"].astype("category")
        VO2_df["Visit"] = VO2_df["Visit"].map(lambda x: x.coerce_to_normal()).astype("category")
        VO2_df["model_type"] = VO2_df["model_type"].astype("category")

        agg_df = (
            VO2_df
            .groupby(['Participant', 'Visit'])
            .agg(
                mse=('sq_error', 'mean'),
                mae=('abs_error', 'mean'),
                n_obs=('sq_error', 'count')
            )
            .reset_index()
        )
        agg_df['rmse'] = np.sqrt(agg_df['mse'])
        agg_df['model_type'] = self.__current_dataframes.VO2_df['model_type'].iloc[0]
        return agg_df

    def _calculate_results(self) -> SetOfModelsResult:
        VO2_rm_corr, VO2_pearson, VO2_bland_altman_global = self._calculate_VO2_results(
            self.__current_dataframes.VO2_df
        )
    
        try:
            VO2_residual_runs_test = self._calculate_residual_runs_test(
                self.__current_dataframes.VO2_df
            )
        except Exception as e:
            print(f"ResultsManager Warning: residual runs test failed: {e}")
            VO2_residual_runs_test = None
    
        try:
            MNG_rm_corr, MNG_bland_altman_global = self._calculate_MNG_results(
                self.__current_dataframes.MNG_df,
                self.__current_dataframes.boxplot_df
            )
        except Exception as e:
            print(f"ResultsManager Warning: MNG results failed: {e}")
            MNG_rm_corr, MNG_bland_altman_global = None, None

        try:
            MNG_rm_corr_averaged, MNG_bland_altman_global_averaged = self._calculate_MNG_results(
                self.__current_dataframes.MNG_df_averaged,
                self.__current_dataframes.boxplot_df_averaged
            )
        except Exception as e:
            print(f"ResultsManager Warning: averaged MNG results failed: {e}")
            MNG_rm_corr_averaged, MNG_bland_altman_global_averaged = None, None

        mse = self.__current_dataframes.VO2_df['mse'].mean()

        results = SetOfModelsResult(
            VO2_rm_corr=VO2_rm_corr,
            VO2_pearson=VO2_pearson,
            VO2_bland_altman_global=VO2_bland_altman_global,
            MNG_rm_corr=MNG_rm_corr,
            MNG_bland_altman_global=MNG_bland_altman_global,
            MNG_rm_corr_averaged=MNG_rm_corr_averaged,
            MNG_bland_altman_global_averaged=MNG_bland_altman_global_averaged,
            mse=mse,
            mae=self.__current_dataframes.VO2_df['mae'].mean(),
            rmse=np.sqrt(mse),
            VO2_residual_runs_test=VO2_residual_runs_test
        )

        # Print results
        self._log(f"Results for {self.__current_paths.model_folder_path}")
        pprint(results)

        return results

    def _calculate_VO2_results(self, VO2_df):

        VO2_df = VO2_df.copy()
        #print("VO2_df columns:", VO2_df.columns.tolist())
        #print(VO2_df.head())

        try:
            rm_corr = Analysis.rm_corr(VO2_df, x='Measured', y='Predicted')        
        except Exception as e:
            print(f"ResultsManager Warning: rm_corr failed: {e}")
            rm_corr = None

        try:
            pearson = Analysis.pearson(VO2_df)
        except Exception as e:
            print(f"ResultsManager Warning: pearson failed: {e}")
            pearson = None

        try:
            bland_altman_global = Analysis.bland_altman_global(VO2_df)
        except Exception as e:
            print(f"ResultsManager Warning: VO2 Bland-Altman failed: {e}")
            bland_altman_global = None

        return rm_corr, pearson, bland_altman_global

    def _get_VO2_df(self):
        VO2_dfs = []
        for participant in self.processed_data:
            for visit in self.processed_data[participant]:
                visit_data = self.processed_data[participant][visit]
                VO2_dfs.append(pd.DataFrame({
                    "model_type": self.__base_path.stem,
                    "Participant": participant,
                    "Visit": visit,
                    "Measured": visit_data.data[self.VO2_column],
                    "Predicted": visit_data.data[self.predictions_column],
                }))

        VO2_df = pd.concat(VO2_dfs)
        VO2_df["mae"] = np.abs(VO2_df["Measured"] - VO2_df["Predicted"])
        VO2_df["mse"] = (VO2_df["Measured"] - VO2_df["Predicted"]) ** 2
        VO2_df["rmse"] = np.sqrt(VO2_df["mse"])

        # ----------------------------------------------------------------
        # <<< NEW: compute and report VO2max (peak VO2) for ramp visits
        # ----------------------------------------------------------------
        self._report_vo2max(VO2_df)

        return VO2_df

    def _report_vo2max(self, VO2_df: pd.DataFrame):
        """
        For every ramp visit, compute per-participant peak measured and
        predicted VO2, print a summary table, and save a CSV to the
        current results folder (vo2max_results.csv).
        """
        # Identify ramp visits.
        ramp_mask = VO2_df["Visit"].apply(lambda v: v.is_ramp())
        ramp_df = VO2_df[ramp_mask].copy()

        if ramp_df.empty:
            return

        rows = []
        for participant, p_df in ramp_df.groupby("Participant"):
            peak_meas = p_df["Measured"].max()
            peak_pred = p_df["Predicted"].max()
            rows.append({
                "Participant": participant,
                "VO2max_measured": round(peak_meas, 1),
                "VO2max_predicted": round(peak_pred, 1),
                "difference": round(peak_pred - peak_meas, 1),
                "abs_error": round(abs(peak_pred - peak_meas), 1),
            })

        vo2max_df = pd.DataFrame(rows).sort_values("Participant")

        # Group summary.
        mean_meas = vo2max_df["VO2max_measured"].mean()
        sd_meas = vo2max_df["VO2max_measured"].std()
        mean_pred = vo2max_df["VO2max_predicted"].mean()
        sd_pred = vo2max_df["VO2max_predicted"].std()
        mean_diff = vo2max_df["difference"].mean()
        sd_diff = vo2max_df["difference"].std()
        mean_ae = vo2max_df["abs_error"].mean()

        # Print table.
        self._log("=" * 60)
        self._log("VO2max summary (ramp visit, peak VO2 per participant)")
        self._log("-" * 60)
        self._log(f"{'Participant':<14} {'Measured':>12} {'Predicted':>12} {'Diff':>8} {'|Diff|':>8}")
        self._log("-" * 60)
        for _, row in vo2max_df.iterrows():
            self._log(
                f"{row['Participant']:<14} "
                f"{row['VO2max_measured']:>12.1f} "
                f"{row['VO2max_predicted']:>12.1f} "
                f"{row['difference']:>8.1f} "
                f"{row['abs_error']:>8.1f}"
            )
        self._log("-" * 60)
        self._log(
            f"{'Group mean':<14} "
            f"{mean_meas:>12.1f} "
            f"{mean_pred:>12.1f} "
            f"{mean_diff:>8.1f}"
        )
        self._log(
            f"{'Group SD':<14} "
            f"{sd_meas:>12.1f} "
            f"{sd_pred:>12.1f} "
            f"{sd_diff:>8.1f}"
        )
        self._log(f"Mean absolute error (VO2max): {mean_ae:.1f} mL min-1")
        self._log("=" * 60)

        # Save CSV.
        if self.__current_paths is not None:
            try:
                out_path = self.__current_paths.results_path / "vo2max_results.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                vo2max_df.to_csv(out_path, index=False)
                self._log(f"VO2max results saved to {out_path}")
            except Exception as e:
                self._warn(f"Could not save vo2max_results.csv: {e}")

    def _prepare_residual_runs_df(self, VO2_df: pd.DataFrame) -> pd.DataFrame:
        runs_df = VO2_df.copy()
        runs_df['Residual'] = runs_df['Predicted'] - runs_df['Measured']
        runs_df['Visit'] = runs_df['Visit'].map(lambda x: x.coerce_to_normal())
        runs_df['_time_index'] = runs_df.index
        runs_df['Participant'] = runs_df['Participant'].astype(str)
        return runs_df

    def _calculate_residual_runs_test(self, VO2_df: pd.DataFrame) -> ResidualRunsTestCollection:
        runs_df = self._prepare_residual_runs_df(VO2_df)

        overall_df = runs_df.sort_values(['Visit', 'Participant', '_time_index'])
        overall = Analysis.residual_runs_test(overall_df['Residual'])

        by_protocol: Dict[str, ResidualRunsTestResult] = {}
        for visit, visit_df in runs_df.groupby('Visit', sort=False):
            visit_name = str(visit)
            visit_df = visit_df.sort_values(['Participant', '_time_index'])
            by_protocol[visit_name] = Analysis.residual_runs_test(visit_df['Residual'])

        by_participant_protocol: Dict[str, ResidualRunsTestResult] = {}
        group_cols = ['Participant', 'Visit']
        for (participant, visit), group_df in runs_df.groupby(group_cols, sort=False):
            group_df = group_df.sort_values('_time_index')
            key = f'{participant}__{visit}'
            by_participant_protocol[key] = Analysis.residual_runs_test(group_df['Residual'])

        return ResidualRunsTestCollection(
            overall=overall,
            by_protocol=by_protocol,
            by_participant_protocol=by_participant_protocol
        )

    @staticmethod
    def _runs_test_result_to_dict(result: Optional[ResidualRunsTestResult]) -> Dict[str, object]:
        if result is None:
            return {}
        return asdict(result)

    def _save_residual_runs_test_csv(self, runs_test: Optional[ResidualRunsTestCollection]):
        if runs_test is None or self.__current_paths is None:
            return

        rows = []

        def add_row(scope: str, result: ResidualRunsTestResult, participant=None, visit=None):
            row = {
                'model_name': self.__model.name,
                'result_folder': str(self.__current_paths.results_path),
                'scope': scope,
                'participant': participant,
                'visit': visit,
            }
            row.update(self._runs_test_result_to_dict(result))
            rows.append(row)

        add_row('overall', runs_test.overall)

        for visit, result in runs_test.by_protocol.items():
            add_row('protocol', result, visit=visit)

        for key, result in runs_test.by_participant_protocol.items():
            participant, visit = key.split('__', 1)
            add_row('participant_protocol', result, participant=participant, visit=visit)

        runs_df = pd.DataFrame(rows)
        self.__current_paths.results_path.mkdir(parents=True, exist_ok=True)
        runs_df.to_csv(self.__current_paths.results_path / self.__residual_runs_test_file, index=False)

    def _calculate_MNG_results(self, MNG_df_averaged, boxplot_df_averaged):
        if len(MNG_df_averaged) == 0:
            return None, None

        rm_corr = Analysis.rm_corr(MNG_df_averaged)
        bland_altman_global = Analysis.bland_altman_global(MNG_df_averaged)
        return rm_corr, bland_altman_global

    def _get_MNG_dfs(self, processed_data: ProcessedData):
        MNG_dfs = []
        boxplot_dfs = []
        for participant in processed_data:
            for visit in processed_data[participant]:
                if visit.is_ramp():
                    continue

                visit_data = processed_data[participant][visit]

                if self.WR_column not in visit_data.data.columns:
                    return MNG_dfs, boxplot_dfs

                WR = visit_data.data[self.WR_column].array
                VO2_measured = visit_data.data[self.VO2_column].array
                VO2_predicted = visit_data.data[self.predictions_column].array

                MNG_measured = Analysis.mng(WR, VO2_measured, cut_off=self.__cut_off_frequency)
                MNG_predicted = Analysis.mng(WR, VO2_predicted, cut_off=self.__cut_off_frequency)

                visit = visit.coerce_to_normal()

                MNG_dfs.append(pd.DataFrame({
                    'Participant': participant,
                    'Visit': visit,
                    'Measured': MNG_measured,
                    'Predicted': MNG_predicted,
                }, index=[len(MNG_dfs)]))
                boxplot_dfs.append(pd.DataFrame({
                    'Participant': participant,
                    'Visit': visit,
                    'Source': 'Measured',
                    'MNG': MNG_measured,
                }, index=[len(boxplot_dfs)]))
                boxplot_dfs.append(pd.DataFrame({
                    'Participant': participant,
                    'Visit': visit,
                    'Source': 'Predicted',
                    'MNG': MNG_predicted,
                }, index=[len(boxplot_dfs)]))

        if len(MNG_dfs) == 0 or len(boxplot_dfs) == 0:
            empty_mng = pd.DataFrame(columns=['Participant', 'Visit', 'Measured', 'Predicted'])
            empty_box = pd.DataFrame(columns=['Participant', 'Visit', 'Source', 'MNG'])
            return empty_mng, empty_box

        return pd.concat(MNG_dfs), pd.concat(boxplot_dfs)

    def _calculate_shap(self, participant_folder) -> ShapResult:
        model = self._load_model(participant_folder)
        participant = Utils.get_participant_from_participant_folder_path(participant_folder)
        X_train, y_train, X_test, y_test, _ = self._get_fold_data(participant)

        sample_size = min(100, X_train.shape[0])
        explain_set = X_train[np.random.choice(X_train.shape[0], sample_size, replace=False)]
        shap_explainer = shap.GradientExplainer(model, explain_set)
        shap_values = shap_explainer.shap_values(X_test)

        shap_features_mean = np.mean(shap_values, axis=1).squeeze()

        return ShapResult(shap_values, shap_features_mean)
