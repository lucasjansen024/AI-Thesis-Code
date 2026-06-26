from abc import ABC, abstractmethod
from typing import List

from keras.src.callbacks import Callback


class Model(ABC):
    batch_size: int = 32
    random_seed: int = 42

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def learning_rate(self) -> float:
        pass

    @property
    @abstractmethod
    def epochs(self) -> int:
        pass

    @property
    @abstractmethod
    def shuffle(self) -> bool:
        pass

    @abstractmethod
    def create_and_compile(self, input_shape):
        pass

    @property
    @abstractmethod
    def receptive_field(self) -> int:
        pass

    @property
    @abstractmethod
    def callbacks(self) -> List[Callback]:
        return []






