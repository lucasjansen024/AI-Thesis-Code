
import math
from dataclasses import dataclass, field
from typing import List

import keras
import tensorflow as tf
from keras.src.callbacks import Callback
from keras.src.optimizers import Adam

from .model import Model

class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_layer, dropout_rate, triple=False):
        super(ResidualBlock, self).__init__()
        self.triple = triple
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_layer = dilation_layer
        self.dropout_rate = dropout_rate

        self.conv1 = keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate=2 ** self.dilation_layer,
            padding="causal",
            activation=None
        )
        self.norm1 = keras.layers.LayerNormalization()
        self.relu1 = keras.layers.ReLU()
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)

        self.conv2 = keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate=2 ** (self.dilation_layer + 1),
            padding="causal",
            activation=None
        )
        self.norm2 = keras.layers.LayerNormalization()
        self.relu2 = keras.layers.ReLU()
        self.dropout2 = keras.layers.Dropout(self.dropout_rate)

        if self.triple:
            self.conv3 = keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                dilation_rate=2 ** (self.dilation_layer + 2),
                padding="causal",
                activation=None
            )
            self.norm3 = keras.layers.LayerNormalization()
            self.relu3 = keras.layers.ReLU()
            self.dropout3 = keras.layers.Dropout(self.dropout_rate)

        self.projection = keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=1,
            padding="same",
            activation=None
        )

    def call(self, inputs, training=None):
        if len(inputs.shape) == 4:
            inputs = tf.squeeze(inputs)

        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x, training=training)

        if self.triple:
            x = self.conv3(x)
            x = self.norm3(x)
            x = self.relu3(x)
            x = self.dropout3(x, training=training)

        inputs_projected = self.projection(inputs)

        return x + inputs_projected

# Defaults must be defined BEFORE TCNConfig
_default_dropout_rate = 0.2
_default_learning_rate = 0.00005

@dataclass
class TCNConfig:
    filters: int = 16
    kernel_size: int = 7
    dilation_depth: int = 5
    dropout_rate: float = _default_dropout_rate

    learning_rate: float = _default_learning_rate
    epochs: int = 30

    shuffle: bool = False
    callbacks: List[Callback] = field(default_factory=list)

    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * (2 ** self.dilation_depth - 1)


class TCN(keras.Model):
    def __init__(self, config: TCNConfig, input_shape, **kwargs):
        super(TCN, self).__init__(**kwargs)

        self.config = config
        self.model_input_shape = input_shape

        # Validate config type
        if not isinstance(config, TCNConfig):
            raise TypeError(f"config must be an instance of {TCNConfig.__name__}")

        # Save TCN parameters from config
        self.kernel_size = config.kernel_size
        self.dilation_depth = config.dilation_depth
        self.filters = config.filters
        self.dropout_rate = config.dropout_rate

        # Define residual blocks
        self.res_blocks = []
        is_odd = self.dilation_depth % 2 == 1
        for i in range(math.floor(self.dilation_depth / 2)):
            dilation_layer = i * 2
            if is_odd and i != 0:
                dilation_layer += 1
            self.res_blocks.append(
                ResidualBlock(filters=self.filters, kernel_size=self.kernel_size, dilation_layer=dilation_layer,
                              dropout_rate=self.dropout_rate, triple=is_odd and i == 0)
            )

        # Dense layer
        self.dense = keras.layers.Dense(1, activation=None)

        # Define input and output tensors
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for block in self.res_blocks:
            x = block(x)
        x = x[:, -1, :]
        outputs = self.dense(x)

        # Set the inputs and outputs
        self.inputs = inputs
        self.outputs = outputs

    def call(self, inputs, training=None):
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    
        x = inputs
    
        for block in self.res_blocks:
            x = block(x, training=training)
    
        # Use representation at last/current time point instead of flattening whole window
        x = x[:, -1, :]
    
        x = self.dense(x)
        return x

    def get_config(self):
        base_config = super(TCN, self).get_config()
        return {**base_config, 'model_config': self.config.__dict__, 'input_shape': self.model_input_shape}

    @classmethod
    def from_config(cls, config):
        model_config = config.pop('model_config')
        input_shape = config.pop('input_shape')
        base_config = TCNConfig(**model_config)
        return cls(base_config, **config, input_shape=input_shape)


class TCNModel(Model):

    def __init__(self, config: TCNConfig):
        self.config = config

    @property
    def name(self):
        dropout = str(self.config.dropout_rate).replace(".", "")
        name = f"TCN_CLL_L{self.receptive_field}_DO{dropout}"
    
        if self.shuffle:
            name += "_S"
    
        if self.learning_rate != _default_learning_rate:
            lr = str(self.learning_rate).replace(".", "")
            name += f"_LR{lr}"
    
        return name

    @property
    def learning_rate(self):
        return self.config.learning_rate

    @property
    def epochs(self) -> int:
        return self.config.epochs

    def create_and_compile(self, input_shape):
        model = TCN(self.config, input_shape=input_shape)
        model.compile(optimizer=Adam(learning_rate=self.config.learning_rate), loss='mean_squared_error', metrics=['mae'])
        return model



    @property
    def receptive_field(self):
        return self.config.receptive_field()

    @property
    def callbacks(self) -> List[Callback]:
        return self.config.callbacks

    @property
    def shuffle(self) -> bool:
        return self.config.shuffle
