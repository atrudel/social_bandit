import abc

from models.RNN import RNN
from bandit_game.utils import History, Choice


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
        if len(history) == 0:
            return Choice.PARTNER_1
        else:
            out = self.model(history.to_torch())
            if out.item() >= 0.5:
                return Choice.PARTNER_2
            else:
                return Choice.PARTNER_1



