from matplotlib import pyplot as plt

from partner import Partner
from player import Player
from utils import History, Choice


class Game:
    def __init__(self, player: Player, partner1: Partner, partner2: Partner):
        self.player: Player = player
        self.partner1: Partner = partner1
        self.partner2: Partner = partner2
        self.history: History = History()

    def play_trial(self, trial_idx: int, round_idx: int):
        choice: Choice = self.player.make_choice(self.history)
        if choice == Choice.PARTNER_1:
            points = self.partner1.play_trial(trial_idx, round_idx)
        else:
            points = self.partner2.play_trial(trial_idx, round_idx)
        self.history.update(choice, points)

    def visualize_one_round(self, round_idx: int):
        for trial_idx in range(len(self.partner1)):
            self.play_trial(trial_idx, round_idx)
        self.visualize(round_idx)

    def visualize(self, round_idx: int):
        plt.subplot(2, 1, 1)
        plt.title("Bandit trajectories and choices made by the model")
        plt.plot(self.partner1.trajectories[0], label="Bandit 0", color="blue")
        plt.plot(self.partner2.trajectories[0], label="Bandit 1", color="orange")
        choices_colors = ["blue" if choice == 0 else "orange" for choice in self.history.choices]
        plt.scatter(list(range(len(self.history.choices))), self.history.choices, label="choices", color=choices_colors)
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Reward")


        plt.subplot(2, 1, 2)
        plt.plot(self.history.points)
        plt.title("Points earned by the player at each turn")
        plt.ylabel("Points")
        plt.xlabel("Turn")
        plt.tight_layout()
        plt.show()