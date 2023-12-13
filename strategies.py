import abc
import enum
from pathlib import Path
from typing import List

import torch

from RNN import RNN
from game import History


class Choice(enum.Enum):
    PARTNER_1 = "Partner 1"
    PARTNER_2 = "Partner 2"


# Base class
class Strategy(abc.ABC):
    @abc.abstractmethod
    def __call__(self, history: History) -> Choice:
        raise NotImplementedError


class AlternatingStrategy(Strategy):
    def __call__(self, history: History) -> Choice:
        if len(history) % 2 == 0:
            return Choice.PARTNER_1
        else:
            return Choice.PARTNER_2

class RNNStrategy(Strategy):
    def __init__(self, model_path: str):
        self.model = RNN.load_from_checkpoint(model_path)

    def __call__(self, history: History) -> Choice:
        out = self.model(history.to_torch())
        # Todo
        raise NotImplementedError



