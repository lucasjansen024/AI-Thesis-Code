import dataclasses
from typing import List
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.src.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.initializers import glorot_uniform

from .model import Model


@dataclasses.dataclass
class DenseNNConfig:
    learning_rate: float = 0.001
    epochs: int = 300


class DenseNNModel(Model):

    def __init__(self, config: DenseNNConfig):
        self.config = config

    @property
    def name(self):
        return 'DenseNN'

    @property
    def learning_rate(self):
        return self.config.learning_rate

    @property
    def epochs(self):
        return self.config.epochs

    @property
    def shuffle(self) -> bool:
        return True

    def create_and_compile(self, input_shape):
        model = Sequential([
            # First hidden layer
            Dense(128, activation='elu', kernel_initializer=glorot_uniform(), input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),

            # Second hidden layer (same structure as the first)
            Dense(128, activation='elu', kernel_initializer=glorot_uniform()),
            BatchNormalization(),
            Dropout(0.3),

            # Output layer
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.config.learning_rate), loss='mean_squared_error', metrics=['mae'])
        return model

    @property
    def receptive_field(self):
        return 1

    @property
    def callbacks(self) -> List[Callback]:
        early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-5)
        return [early_stopping, reduce_lr]