import keras
from .TCN import TCNConfig, TCNModel
from .DenseNN import DenseNNConfig, DenseNNModel
from .GRU import GRUConfig, GRUModel

TCN_1 = TCNModel(
    TCNConfig(epochs=1)
)

TCN_1_shuffle = TCNModel(
    TCNConfig(epochs=1, shuffle=True)
)

TCN_30 = TCNModel(
    TCNConfig()
)


# TCN thesis models
TCN_SHORT_DO00 = TCNModel(TCNConfig(kernel_size=7, dilation_depth=4, dropout_rate=0.0, epochs=30))
TCN_SHORT_DO02 = TCNModel(TCNConfig(kernel_size=7, dilation_depth=4, dropout_rate=0.2, epochs=30))
TCN_SHORT_DO04 = TCNModel(TCNConfig(kernel_size=7, dilation_depth=4, dropout_rate=0.4, epochs=30))

TCN_BASE_DO00 = TCNModel(TCNConfig(kernel_size=7, dilation_depth=5, dropout_rate=0.0, epochs=30))
TCN_BASE_DO02 = TCNModel(TCNConfig(kernel_size=7, dilation_depth=5, dropout_rate=0.2, epochs=30))
TCN_BASE_DO02_10 = TCNModel(TCNConfig(kernel_size=7, dilation_depth=5, dropout_rate=0.2, epochs=10))
TCN_BASE_DO02_5 = TCNModel(TCNConfig(kernel_size=7, dilation_depth=5, dropout_rate=0.2, epochs=5))

TCN_BASE_DO04 = TCNModel(TCNConfig(kernel_size=7, dilation_depth=5, dropout_rate=0.4, epochs=30))

TCN_LONG_DO00 = TCNModel(TCNConfig(kernel_size=11, dilation_depth=5, dropout_rate=0.0, epochs=30))
TCN_LONG_DO02 = TCNModel(TCNConfig(kernel_size=11, dilation_depth=5, dropout_rate=0.2, epochs=30))
TCN_LONG_DO04 = TCNModel(TCNConfig(kernel_size=11, dilation_depth=5, dropout_rate=0.4, epochs=30))


GRU_SHORT_DO00 = GRUModel(GRUConfig(receptive_field=91, dropout_rate=0.0, epochs=30))
GRU_SHORT_DO02 = GRUModel(GRUConfig(receptive_field=91, dropout_rate=0.2, epochs=30))
GRU_SHORT_DO04 = GRUModel(GRUConfig(receptive_field=91, dropout_rate=0.4, epochs=30))

GRU_BASE_DO00 = GRUModel(GRUConfig(receptive_field=187, dropout_rate=0.0, epochs=30))
GRU_BASE_DO02 = GRUModel(GRUConfig(receptive_field=187, dropout_rate=0.2, epochs=30))
GRU_BASE_DO04 = GRUModel(GRUConfig(receptive_field=187, dropout_rate=0.4, epochs=30))

GRU_LONG_DO00 = GRUModel(GRUConfig(receptive_field=311, dropout_rate=0.0, epochs=30))
GRU_LONG_DO02 = GRUModel(GRUConfig(receptive_field=311, dropout_rate=0.2, epochs=30))
GRU_LONG_DO04 = GRUModel(GRUConfig(receptive_field=311, dropout_rate=0.4, epochs=30))

TCN_30_shuffle = TCNModel(
    TCNConfig(shuffle=True)
)

TCN_100 = TCNModel(
    TCNConfig(epochs=100)
)

TCN_1_shuffle_lr_0005 = TCNModel(
    TCNConfig(shuffle=True, learning_rate=0.0005, epochs=1)
)

TCN_2_shuffle_lr_0005 = TCNModel(
    TCNConfig(shuffle=True, learning_rate=0.0005, epochs=2)
)

TCN_3_shuffle_lr_0005 = TCNModel(
    TCNConfig(shuffle=True, learning_rate=0.0005, epochs=3)
)

TCN_1_shuffle_lr_0005_do_25 = TCNModel(
    TCNConfig(shuffle=True, learning_rate=0.0005, epochs=1, dropout_rate=0.25)
)

TCN_30_shuffle_lr_0005 = TCNModel(
    TCNConfig(shuffle=True, learning_rate=0.0005)
)

DenseNN = DenseNNModel(DenseNNConfig())

# --- GRU models (fair comparison defaults) ---
# GRUConfig defaults match default TCNConfig for window length & training settings.
GRU_30 = GRUModel(GRUConfig())                 # shuffle=False by default
GRU_30_shuffle = GRUModel(GRUConfig(shuffle=True))
