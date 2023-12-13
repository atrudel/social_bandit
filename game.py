from typing import List

import torch
from matplotlib import pyplot as plt

from partner import Partner
from player import Player
from strategies import Choice
from tqdm import trange


class History:
    def __init__(self):
        self.choices: List[Choice] = []
        self.points: List[int] = []

    def update(self, choice: Choice, points: int):
        self.choices.append(choice)
        self.points.append(points)

    def to_torch(self) -> torch.Tensor:
        choices = torch.tensor(self.choices).reshape(1, -1)
        points = torch.tensor(self.points).reshape(1, -1)
        return torch.stack([choices, points], dim=2)

    def __len__(self) -> int:
        return len(self.choices)

class Game:
    def __init__(self, player: Player, partner1: Partner, partner2: Partner):
        self.player: Player = player
        self.partner1: Partner = partner1
        self.partner2: Partner = partner2
        self.history: History = History()
        self.turn = 0

    def play_trial(self):
        choice: Choice = self.player.make_choice(self.turn)
        if choice == Choice.PARTNER_1:
            points = self.partner1.play_trial(self.turn)
            self.partner2.pass_turn(self.turn)
        else:
            points = self.partner2.play_trial(self.turn)
            self.partner1.pass_turn(self.turn)
        self.turn += 1
        self.history.update(choice, points)

    def play_n_turns(self, n: int):
        for round in trange(n):
            self.play_trial()

    def visualize(self):
        plt.plot(self.history.points)
        plt.title("Points earned by the player in each turn")
        plt.ylabel("Points")
        plt.xlabel("Turn")
        plt.show()