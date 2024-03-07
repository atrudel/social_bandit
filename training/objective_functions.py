import abc

import torch
from torch import Tensor
import scipy


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
        """Computes the cumulative sum of discounted future rewards for each time step (called return)"""
        reversed_rewards = rewards.flip(1).numpy()
        reversed_returns = scipy.signal.lfilter([1], [1, -self.discount_factor], reversed_rewards, axis=1)
        return torch.tensor(reversed_returns).flip(1).float()
