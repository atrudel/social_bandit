from strategies import Strategy
from bandit_game.utils import Choice, History


class Player:
    def __init__(self, strategy: Strategy):
        self.strategy: Strategy = strategy

    def make_choice(self, history: History) -> Choice:
        choice: Choice = self.strategy(history)
        return choice
