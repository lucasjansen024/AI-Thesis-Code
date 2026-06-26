from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import keras
from keras.src.callbacks import Callback
from keras.src.optimizers import Adam

from .model import Model
from .TCN import TCNConfig


# Match the default TCN window length automatically (fair input comparison).
_DEFAULT_RECEPTIVE_FIELD = TCNConfig().receptive_field()

# Match the default TCN hyperparameters for fair comparison.
_DEFAULT_LEARNING_RATE = 0.00005
_DEFAULT_DROPOUT_RATE = 0.2


@dataclass
class GRUConfig:
    """
    IMPORTANT: In this project, `receptive_field` == window length L used by DataManager.
    Keeping this equal to TCN's receptive_field ensures identical model inputs.
    """
    receptive_field: int = _DEFAULT_RECEPTIVE_FIELD

    units: int = 32
    num_layers: int = 1
    bidirectional: bool = False

    dropout_rate: float = _DEFAULT_DROPOUT_RATE
    recurrent_dropout_rate: float = 0.0

    learning_rate: float = _DEFAULT_LEARNING_RATE
    epochs: int = 30
    shuffle: bool = False

    # Fairness note: TCN by default uses no callbacks, so GRU defaults to none too.
    callbacks: List[Callback] = field(default_factory=list)


class GRUModel(Model):
    def __init__(self, config: GRUConfig):
        self.config = config

    @property
    def name(self) -> str:
        name = f"GRU_L{self.config.receptive_field}_U{self.config.units}_N{self.config.num_layers}"
        
        if self.config.bidirectional:
            name += "_Bi"
        if self.shuffle:
            name += "_S"
        if self.learning_rate != _DEFAULT_LEARNING_RATE:
            name += f"_LR{self.learning_rate}"
        if self.config.dropout_rate != _DEFAULT_DROPOUT_RATE:
            name += f"_D{self.config.dropout_rate}"
            
        return name

    @property
    def learning_rate(self) -> float:
        return self.config.learning_rate

    @property
    def epochs(self) -> int:
        return self.config.epochs

    @property
    def shuffle(self) -> bool:
        return self.config.shuffle

    @property
    def receptive_field(self) -> int:
        # DataManager uses this as the window length when splitting sequences
        return int(self.config.receptive_field)

    @property
    def callbacks(self) -> List[Callback]:
        # For fair comparison, default is no callbacks (same as default TCN setup).
        return list(self.config.callbacks)

    def create_and_compile(self, input_shape):
        """
        input_shape = (timesteps, features) from Trainer
        Output: single VO2 prediction (regression).
        """
        inputs = keras.Input(shape=input_shape)
        x = inputs

        # Stack GRU layers. All but the last return sequences.
        n_layers = max(int(self.config.num_layers), 1)
        for i in range(n_layers):
            return_sequences = i < (n_layers - 1)
            layer = keras.layers.GRU(
                units=self.config.units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout_rate,
            )
            if self.config.bidirectional:
                x = keras.layers.Bidirectional(layer)(x)
            else:
                x = layer(x)

        # Small regression head
        x = keras.layers.Dense(self.config.units, activation="elu")(x)
        outputs = keras.layers.Dense(1, activation=None)(x)

        model = keras.Model(inputs, outputs, name=self.name)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mean_squared_error",
            metrics=["mae"],
        )
        return model
