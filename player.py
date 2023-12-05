from typing import List

from strategies import Strategy, Choice


class Player:
    def __init__(self, strategy: Strategy):
        self.history: List[int] = []
        self.strategy: Strategy = strategy

    def choose_partner(self, turn: int) -> Choice:
        choice: Choice = self.strategy(self.history, turn)
        return choice
