import abc
import enum
from pathlib import Path
from typing import List


class Choice(enum.Enum):
    PARTNER_1 = "Partner 1"
    PARTNER_2 = "Partner 2"


# Base class
class Strategy(abc.ABC):
    @abc.abstractmethod
    def __call__(self, history: List[int], turn: int) -> Choice:
        raise NotImplementedError


class DummyStrategy(Strategy):
    def __call__(self, history: List[int], turn: int) -> Choice:
        if turn % 2 == 0:
            return Choice.PARTNER_1
        else:
            return Choice.PARTNER_2

class RNNStrategy(Strategy):
    def __init__(self, model_path: str):
        path = Path(model_path)


