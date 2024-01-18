from typing import List

from strategies import Strategy
from utils import Choice, History


class Player:
    def __init__(self, strategy: Strategy):
        self.strategy: Strategy = strategy

    def make_choice(self, history: History) -> Choice:
        choice: Choice = self.strategy(history)
        return choice
