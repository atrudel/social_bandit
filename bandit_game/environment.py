from typing import Tuple, List

import torch
from torch import Tensor

import config
from bandit_game.chooser import Chooser
from bandit_game.partner import Partner
from bandit_game.utils import History


class Env:
    def __init__(self,
                 chooser: Chooser,
                 partner_0: Partner,
                 partner_1: Partner,
                 n_trials: int = config.N_TRIALS
                 ):
        self.chooser: Chooser = chooser
        self.partner_0: Partner = partner_0
        self.partner_1: Partner = partner_1
        self.n_trials: int = n_trials
        self.played_trials: int = 0
        self.sharable_points: List[Tensor] = []

    def play_full_episode(self, trajectories) -> Tuple[History, History, History]:
        batch_size = trajectories.shape[0]
        self._reset()
        self.chooser.reset()
        self.partner_0.reset(trajectories[:, 0])
        self.partner_1.reset(trajectories[:, 1])
        for trial in range(self.n_trials):
            self._step(batch_size)
        return self.chooser.history, self.partner_0.history, self.partner_1.history

    def _step(self, batch_size: int) -> None:
        # Both partners are asked how many points they would give if they were chosen
        sharable_points_0 = self.partner_0.play_trial()  # [B, 1]
        sharable_points_1 = self.partner_1.play_trial()
        sharable_points = torch.cat([  # [B, 1, 2] = [Batch, trial, partner]
            sharable_points_0.unsqueeze(2),
            sharable_points_1.unsqueeze(2)
        ], dim=2)

        # Chooser chooses a partner to play with
        choice: Tensor = self.chooser.make_choice(batch_size)  # [B, 1]

        # Compute reward for all players
        chooser_reward = torch.gather(sharable_points, 2, choice.unsqueeze(1)).squeeze(1)  # [B, 1]
        partner_0_reward = (1 - sharable_points_0) * (1 - choice)
        partner_1_reward = (1 - sharable_points_1) * choice

        # Update all players with trial outcome
        self.chooser.update_reward(chooser_reward)
        self.partner_0.update_record(choice, partner_0_reward)
        self.partner_1.update_record(choice, partner_1_reward)

        # Update record of points each partner was willing to share
        # This will be useful to calculate the excess reward
        self.sharable_points.append(sharable_points)

    def _reset(self) -> None:
        self.played_trials = 0
        self.sharable_points = []

    def get_target_choices(self):
        """Compute what partner would have been optimal to choose at every trial given how many each of them was willing to share"""
        sharing_history = torch.cat(self.sharable_points, dim=1)
        targets = torch.argmax(sharing_history, dim=2).detach().float()
        return targets


    # def visualize_one_round(self, round_idx: int):
    #     for trial_idx in range(len(self.partner_0)):
    #         self.step(trial_idx, round_idx)
    #     self._visualize(round_idx)
    #
    # def _visualize(self, round_idx: int):
    #     plt.subplot(2, 1, 1)
    #     plt.title("Bandit trajectories and choices made by the model")
    #     plt.plot(self.partner_0.trajectories[0], label="Bandit 0", color="tab:blue")
    #     plt.plot(self.partner_0.trajectories[0], label="Bandit 1", color="tab:orange")
    #     choices_colors = ["tab:blue" if choice == 0 else "tab:orange" for choice in self.history.choices]
    #     plt.scatter(list(range(len(self.history.choices))), self.history.choices, label="choices", color=choices_colors)
    #     plt.legend()
    #     plt.xlabel("Time step")
    #     plt.ylabel("Reward")
    #
    #
    #     plt.subplot(2, 1, 2)
    #     plt.plot(self.history.points)
    #     plt.title("Points earned by the player at each turn")
    #     plt.ylabel("Points")
    #     plt.xlabel("Turn")
    #     plt.tight_layout()
    #     plt.show()