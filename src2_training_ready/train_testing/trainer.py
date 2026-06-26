import keras
import pickle
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from src2_training_ready.models.TCN import TCN
from ..data_types.base import TrainingData, Visit
from ..models.models import TCN_30
from ..models.model import Model
from ..utils import Utils
from .data_manager import DataManager


class Trainer:
    def __init__(
            self,
            base_path: Path,
            model: Model = TCN_30,
            logging=True
    ):
        self.__base_path: Path = base_path
        self.__current_model_path: Path = None
        self.__model = model
        self.__logging = logging
        tf.random.set_seed(model.random_seed)

    def _log(self, message: str):
        if self.__logging:
            print(f"{datetime.now().strftime('%H:%M:%S')} Trainer Info: {message}")

    def train_LOOCV(
            self,
            data_manager: DataManager,
            train_visits: list[Visit] | None = None,
            test_visits: list[Visit] | None = None,
    ) -> str:
        if not data_manager.processed:
            raise ValueError("DataManager must be processed before training")

        model_folder = self.__create_model_folder()
        self._log(f"Training model with LOOCV, saving to {model_folder}")

        participants = list(data_manager.processed_data.keys())
        for participant in participants:
            self._log(f"Training model for {participant}")
            X_train, y_train, X_test, y_test, fold_scalers = data_manager.get_fold_data(
                participant,
                train_visits=train_visits,
                test_visits=test_visits,
            )
            model = self.__create_model_from_arrays(X_train)
            p_folder_path = self.__current_model_path / participant
            p_folder_path.mkdir(parents=True, exist_ok=True)
            self.__train_model(p_folder_path, model, X_train, X_test, y_train, y_test)
            pickle.dump(fold_scalers, open(p_folder_path / 'fold_scalars.pkl', 'wb'))
        return model_folder

    def __create_model_folder(self):
        model_folder = datetime.now().strftime("%y%m%d_%H%M%S")
        self.__current_model_path = self.__base_path / model_folder
        Path(self.__current_model_path).mkdir(parents=True, exist_ok=True)
        return model_folder

    def __create_model(self, training_data: TrainingData):
        first_visit_data = Utils.get_first_visit_data(training_data)
        input_shape = (first_visit_data.X.shape[1], first_visit_data.X.shape[2])
        return self.__model.create_and_compile(input_shape)

    def __create_model_from_arrays(self, X_train):
        input_shape = (X_train.shape[1], X_train.shape[2])
        return self.__model.create_and_compile(input_shape)

    def __train_model(self, p_folder_path: Path, model, X_train, X_test, y_train, y_test):
        Path(p_folder_path).mkdir(parents=True, exist_ok=True)
    
        checkpoint_path = str(p_folder_path / "best_model.weights.h5")
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        )
    
        history = model.fit(
            X_train,
            y_train,
            epochs=self.__model.epochs,
            batch_size=self.__model.batch_size,
            callbacks=[checkpoint, *self.__model.callbacks],
            shuffle=self.__model.shuffle,
            verbose=0,
        )
    
        model.load_weights(checkpoint_path)
    
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        self._log(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
    
        model.save(str(p_folder_path / "final_model.keras"))
        pickle.dump(history.history, open(p_folder_path / "history.pkl", "wb"))



