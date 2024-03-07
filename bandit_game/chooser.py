from __future__ import annotations

import abc
from typing import Optional

import torch
from torch import Tensor, nn

from bandit_game.utils import History
from models.rnn_chooser import RNNforBinaryAction


### Class for the Chooser Player

class Chooser:
    def __init__(self, policy: ChooserPolicy):
        self.policy: ChooserPolicy = policy
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


###############################################
### Classes for the policies used by the Chooser

# Base class
class ChooserPolicy(abc.ABC):
    @abc.abstractmethod
    def __call__(self, last_action: Tensor, last_reward: Tensor) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, trajectory: Optional[Tensor] = None) -> None:
        raise NotImplementedError


# RNN_based policy
class RNNChooserPolicy(ChooserPolicy):
    def __init__(self, model: nn.Module):
        self.model = model
        self.hidden_state: Optional[Tensor] = None

    def __call__(self, last_action: Tensor, last_reward: Tensor) -> Tensor:
        rnn_input = torch.stack([last_action, last_reward], dim=2)
        out_prob, hidden_state = self.model(rnn_input, self.hidden_state)
        self.hidden_state = hidden_state
        return out_prob

    def reset(self, trajectory: Optional[Tensor] = None) -> None:
        # The RNN policy doesn't rely on a pre-determined trajectory
        self.hidden_state = None

    @classmethod
    def load(cls, model_path: str) -> RNNChooserPolicy:
        policy = RNNChooserPolicy(0, 0)
        policy.model = RNNforBinaryAction.load(save_path=model_path)
        return policy
