import abc

import torch
from torch import Tensor

class ObjectiveFunction(abc.ABC):
    """Abstract class for all objective functions."""
    @abc.abstractmethod
    def compute_loss(self, probs, actions: Tensor, rewards: Tensor) -> Tensor:
        raise NotImplementedError


class RewardObjectiveFunction(ObjectiveFunction):
    def __init__(self, discount_factor: float):
        self.discount_factor: float = discount_factor
    def compute_loss(self, probs: Tensor, actions: Tensor, rewards: Tensor) -> Tensor:
        returns: Tensor = self._compute_returns(rewards)
        action_probs = probs * actions + (1 - probs) * (1 - actions)
        policy_gradient = (torch.log(action_probs) * returns).sum(dim=1)
        loss = -policy_gradient
        return loss.mean()

    def _compute_returns(self, rewards: Tensor) -> Tensor:
        returns = torch.zeros_like(rewards)
        length = rewards.shape[1]
        for t in range(length):
            returns[:, t] = sum([rewards[:, t+j] * self.discount_factor**j for j in range(length-t)])
        return returns
