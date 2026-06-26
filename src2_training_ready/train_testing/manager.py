import dataclasses
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
import pickle
import pandas as pd

from ..constants import *
from ..data_types.base import Participant
from ..processing.processor import Processor
from .data_manager import DataManager
from .plotter import Plotter
from .results_manager import ResultsManager, Dataframes
from ..analysis import Analysis
from ..models.models import TCN_30
from ..models.model import Model
from ..utils import Utils
from .trainer import Trainer
from ..data_types.results import SetOfModelsResult, BlandAltmanResult, RMCorrResult


@dataclasses.dataclass
class ManagerConfig:
    visit_selection: VisitSelection = VisitSelection.SUBMAX_PLUS_RAMP
    participant_selection: ParticipantSelection = ParticipantSelection.LUCAS_ALL
    model: Model = TCN_30
    train_visit_selection: VisitSelection | None = None
    test_visit_selection: VisitSelection | None = None


class Manager:

    def __init__(
            self,
            # processor: Processor,
            config: ManagerConfig,
            logging=True,
            plot_MNG=False
    ):
        # self.__processor = processor
        self.__participant_selection = config.participant_selection
        self.__visit_selection = config.visit_selection
        self.__train_visit_selection = config.train_visit_selection or config.visit_selection
        self.__test_visit_selection = config.test_visit_selection or config.visit_selection
        self.__train_visits = list(self.__train_visit_selection.value)
        self.__test_visits = list(self.__test_visit_selection.value)
        self.model = config.model
        self.__logging = logging

        self.__plot_MNG = plot_MNG
        self.__scalars_file = "scalars.pkl"

        self.processor = None
        self.data_manager = None
        self.plotter = None
        self.trainer = None
        self.results_manager = None

    def set_processor(self, processor: Processor):
        self.processor = processor
        self.__set_base_folder()

        self.data_manager = DataManager(
            required_files=processor.get_required_files(),
            visit_selection=self.__visit_selection,
            participant_selection=self.__participant_selection,
            model=self.model,
            logging=self.__logging
        )
        self.plotter = Plotter(participant_selection=self.__participant_selection)
        self.trainer = Trainer(
            base_path=self.base_path,
            model=self.model,
            logging=self.__logging
        )
        self.results_manager = ResultsManager(
            base_path=self.base_path,
            logging=self.__logging,
            model=self.model,
            train_visits=self.__train_visits,
            test_visits=self.__test_visits,
        )

    def _log(self, message: str):
        if self.__logging:
            print(f"{datetime.now().strftime('%H:%M:%S:%f')} Manager Info: {message}")

    def __set_base_folder(self):
        visit_short = {
            "SUBMAX_PLUS_RAMP": "SPR",
            "ALL_PROTOCOLS": "ALL",
            "ONLY_RAMP": "R",
            "ONLY_PRBS": "P",
            "ONLY_INTERVAL": "I",
            "ONLY_STEP": "S",
            "ALL_EXCEPT_RAMP": "NO_R",
            "ALL_EXCEPT_PRBS": "NO_P",
            "ALL_EXCEPT_INTERVAL": "NO_I",
            "ALL_EXCEPT_STEP": "NO_S",
            "RAMP_PRBS": "RP",
            "RAMP_INTERVAL": "RI",
            "RAMP_STEP": "RS",
            "PRBS_INTERVAL": "PI",
            "PRBS_STEP": "PS",
            "INTERVAL_STEP": "IS",
        }

        participant_short = {
            "COMPLETE_4_PROTOCOLS": "C4",
            "RAMP": "R",
            "PRBS": "P",
            "INTERVAL": "I",
            "STEP": "S",
            "RAMP_PRBS": "RP",
            "RAMP_INTERVAL": "RI",
            "RAMP_STEP": "RS",
            "PRBS_INTERVAL": "PI",
            "PRBS_STEP": "PS",
            "INTERVAL_STEP": "IS",
            "PRBS_INTERVAL_STEP": "PIS",
        }

        train_name = visit_short.get(self.__train_visit_selection.name, self.__train_visit_selection.name)
        test_name = visit_short.get(self.__test_visit_selection.name, self.__test_visit_selection.name)
        participant_name = participant_short.get(self.__participant_selection.name, self.__participant_selection.name)

        self.base_path: Path = (
            MODEL_PATH
            / self.model.name
            / f"{train_name}_to_{test_name}"
            / participant_name
            / f"e{self.model.epochs}"
            / self.processor.get_folder_name()
        )

        self._log(f"Base path set to {self.base_path}")


    @staticmethod
    def get_participant_selection_from_base_path(base_path: Path) -> ParticipantSelection:
        for participant_selection in ParticipantSelection:
            if participant_selection.name in base_path.name:
                return participant_selection

    @staticmethod
    def get_visit_selection_from_base_path(base_path: Path) -> VisitSelection:
        for visit_selection in VisitSelection:
            if visit_selection.name in base_path.name:
                return visit_selection

    def __check_base_folder_exists(self):
        exists = os.path.exists(self.base_path)
        if not exists:
            self._log(f"Base folder {self.base_path} does not exist")
        return exists

    def set_data_for_results_manager(self):
        if not self.data_manager.processed:
            self.process_prepared_data()

        # ResultsManager now uses fold-wise scaling from DataManager.get_fold_data(...),
        # so we do not need to precompute global scalers/training arrays here.
        self.results_manager.set_data(
            processed_data=self.data_manager.processed_data_with_raw_VO2,
            training_data=None,
            scalars=None,
            data_manager=self.data_manager,
        )

    def get_common_folder(self) -> Path:
        return self.base_path / COMMON_FOLDER

    def get_best_folder(self) -> Path:
        return self.base_path / BEST_FOLDER

    def load_prepared_data(self):
        self.data_manager.load_prepared_data()

    def process_prepared_data(self):
        self.data_manager.process_data(self.processor)

        if not os.path.exists(self.get_common_folder()):
            self.plot_processed_data()

    def plot_processed_data(self):
        if not self.data_manager.processed:
            self.data_manager.process_data(self.processor)

        self._log("Plotting processed data")
        for participant in self.data_manager.processed_data:
            for visit in self.data_manager.processed_data[participant]:
                processed_data = self.data_manager.processed_data[participant][visit]
                self.plotter.plot_processed_data(
                    processed_data,
                    self.get_common_folder(),
                    self.model.receptive_field
                )

    def prepare_training_data(self):
        raise RuntimeError(
            "prepare_training_data() uses global scalers fit on all participants (leakage risk). "
            "Use train_LOOCV() + ResultsManager with DataManager.get_fold_data() for fold-wise scaling."
        )
        if not self.data_manager.processed:
            self.process_prepared_data()

        self.data_manager.prepare_training_data()

        if not os.path.exists(self.get_common_folder() / self.__scalars_file):
            self.save_scalars()
        # else:
        #     scalars = pickle.load(open(self.get_common_folder() / self.__scalars_file, 'rb'))
        #     for scalar in scalars:
        #         print(f"Scalar column: {scalar}")

    def save_scalars(self):
        self._log(f"Saving Scalars to {self.get_common_folder()}")
        pickle.dump(self.data_manager.scalars, open(self.get_common_folder() / self.__scalars_file, 'wb'))

    def train_LOOCV(self) -> str:
        if not self.data_manager.processed:
            self.process_prepared_data()

        return self.trainer.train_LOOCV(
            self.data_manager,
            train_visits=self.__train_visits,
            test_visits=self.__test_visits,
        )

    def train_test(
            self,
            exclude_participants: List[Participant] = [],
            times: int = 1,
            hard_cap=True
    ):
        trained_models = self.results_manager.get_amount_of_trained_models(exclude_participants)

        if hard_cap:
            times = times - trained_models

        if times > 0:
            for i in range(times):
                model_folder = self.train_LOOCV()
                self.test(model_folder, exclude_participants)

    def test(
            self,
            model: str | int,  # model is either specific model folder, i.e: 2025-02-06_13.19, or index in the folder
            exclude_participants: List[Participant] = [],
            plot=True,
            force=False
    ) -> Tuple[SetOfModelsResult, Dataframes | None] | None:
        if not self.__check_base_folder_exists():
            return None

        self.set_data_for_results_manager()
        if type(model) is int:
            model = self.results_manager.get_model_folder_from_index(model)

        if model != BEST_FOLDER and not force:
            existing_results = self.load_results(model, exclude_participants)
            if existing_results:
                return existing_results, None

        self.results_manager.prepare_data(model, exclude_participants)

        if plot:
            self._plot_predictions(model)

        results = self.results_manager.test_folder(model, exclude_participants, force)
        dataframes = self.results_manager.get_dataframes(model, exclude_participants)

        if plot:
            self._plot_results(
                results,
                dataframes,
                self.results_manager.get_results_path(model)
            )

        return results, dataframes

    def _plot_results(self, result: SetOfModelsResult, dataframes: Dataframes, save_path: Path):
        self._log("Plotting results")
        self._plot_VO2_results(result, dataframes.VO2_df, save_path)

        if self.__plot_MNG:
            self._plot_MNG_results(result, dataframes, save_path)

    def _plot_VO2_results(self, result: SetOfModelsResult, VO2_df: pd.DataFrame, save_path: Path):
        if result.VO2_rm_corr is not None and result.VO2_pearson is not None:
            self.plotter.plot_VO2_correlation(VO2_df, result.VO2_rm_corr, result.VO2_pearson, save_path)

        self.plotter.plot_VO2_residuals(VO2_df, save_path)

        if result.VO2_bland_altman_global is not None:
            participant_VO2_bland_altman_results = {}
            for participant in VO2_df['Participant'].unique():
                participant_df = VO2_df[VO2_df['Participant'] == participant]
                bland_altman_result = Analysis.bland_altman(participant_df['Measured'], participant_df['Predicted'])
                participant_VO2_bland_altman_results[participant] = bland_altman_result

            self.plotter.plot_VO2_bland_altman_global(
                participant_VO2_bland_altman_results,
                result.VO2_bland_altman_global,
                save_path
            )
    def _plot_MNG_results(self, result: SetOfModelsResult, dataframes: Dataframes, save_path: Path):
        self._plot_MNG_result(
            dataframes.MNG_df,
            dataframes.boxplot_df,
            result.MNG_bland_altman_global,
            result.MNG_rm_corr,
            save_path,
            "both"
        )
        self._plot_MNG_result(
            dataframes.MNG_df_averaged,
            dataframes.boxplot_df_averaged,
            result.MNG_bland_altman_global_averaged,
            result.MNG_rm_corr_averaged,
            save_path,
            "averaged"
        )


    def _plot_MNG_result(
            self,
            MNG_df: pd.DataFrame,
            boxplot_df: pd.DataFrame,
            bland_altman_result: BlandAltmanResult,
            rm_corr_result: RMCorrResult,
            save_path: Path,
            identifier: str = ''
    ):
        participant_MNG_bland_altman_results = {}
        for participant in MNG_df['Participant'].unique():
            participant_df = MNG_df[MNG_df['Participant'] == participant]
            bland_altman = Analysis.bland_altman(participant_df['Measured'], participant_df['Predicted'])
            visit = participant_df['Visit']
            participant_MNG_bland_altman_results[participant] = (bland_altman, visit)

        self.plotter.plot_MNG_bland_altman_global(participant_MNG_bland_altman_results, bland_altman_result,
                                                  save_path, identifier=identifier)
        self.plotter.plot_mng_boxplot(boxplot_df, save_path, identifier=identifier)
        self.plotter.plot_mng_rm_corr(MNG_df, rm_corr_result, save_path=save_path,identifier=identifier)

    def _plot_predictions(self, model_folder: str):
        results_data = self.results_manager.get_processed_data()
        model_folder_path = self.results_manager.get_model_folder_path(model_folder)

        if model_folder != BEST_FOLDER:
            first_visit_data = Utils.get_first_visit_data(results_data)
            first_visit_path = model_folder_path / first_visit_data.participant / first_visit_data.visit
            if os.path.exists(first_visit_path / self.plotter.VO2_predictions_file):
                self._log(f"Predictions already plotted for {model_folder}, continuing")
                return

        self._log(f"Plotting predictions for {model_folder}")
        for participant in results_data:
            for visit in results_data[participant]:
                visit_path = model_folder_path / participant / visit
                if os.path.exists(visit_path / self.plotter.VO2_predictions_file):
                    continue

                result_visit_data = results_data[participant][visit]
                self.plotter.plot_prediction_results(result_visit_data, visit_path)

    def test_all(
            self,
            exclude_participants: List[Participant] = [],
            plot=True,
            force=False
    ) -> Dict[str, Tuple[SetOfModelsResult, Dataframes]]:
        results: Dict[str, Tuple[SetOfModelsResult, Dataframes]] = {}
        if not self.__check_base_folder_exists():
            return results

        self.set_data_for_results_manager()

        for model_folder in self.results_manager.get_all_model_folders():
            results[model_folder] = self.test(model_folder, exclude_participants, plot, force)

        return results

    def test_best(
            self,
            exclude_participants: List[Participant] = [],
            plot=True
    ) -> Tuple[SetOfModelsResult, Dataframes]:
        return self.test(BEST_FOLDER, exclude_participants, plot)

    def load_results(
            self,
            model: str | int,  # model is either specific model folder, i.e: 2025-02-06_13.19, or index in the folder
            exclude_participants: List[Participant] = [],
    ) -> SetOfModelsResult | None:
        if not self.__check_base_folder_exists():
            return None

        if type(model) is int:
            model = self.results_manager.get_model_folder_from_index(model)

        return self.results_manager.load_results(model, exclude_participants)

    def load_all(
            self,
            include_best=True,
            exclude_participants: List[Participant] = []
    ) -> Dict[str, SetOfModelsResult | None]:
        if not self.__check_base_folder_exists():
            return {}

        results = self.results_manager.load_all(include_best, exclude_participants)

        return results

    def load_aggregated_df(self) -> pd.DataFrame | None:
        return self.results_manager.load_aggregated_df()

    def mean_results(
            self,
            exclude_participants: List[Participant] = []
    ) -> SetOfModelsResult | None:
        if not self.__check_base_folder_exists():
            return None

        return self.results_manager.mean_results(exclude_participants)
