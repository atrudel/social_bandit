from __future__ import annotations

import abc
from typing import Optional

import torch
from torch import Tensor

from social_bandit.game.utils import History


### Class for the Chooser Player

class Chooser:
    def __init__(self, policy):
        self.policy = policy
        self.history: History = History()

    def make_choice(self, batch_size: int) -> Tensor:
        # On first trial play randomly
        if len(self.history) == 0:
            prob: Tensor = torch.full(size=(batch_size, 1), fill_value=0.5, dtype=torch.float)
        else:
            last_choice, last_reward = self.history.get_last_trial()
            prob: Tensor = self.policy(last_choice, last_reward)
        choice = torch.bernoulli(prob).long()  # Todo: Reparametrization
        self.history.add_choice_and_prob(choice, prob)
        return choice

    def update_trial(self, choice: Tensor, reward: Tensor):
        self.history.update(choice, reward)

    def update_reward(self, reward: Tensor):
        self.history.add_points(reward)

    def reset(self, trajectory: Optional[Tensor] = None):
        self.history = History()
        self.policy.reset(trajectory)

    def __repr__(self) -> str:
        return f"Chooser(policy={str(self.policy)})"
