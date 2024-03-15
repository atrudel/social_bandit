import abc

import scipy
import torch
from torch import Tensor


class ObjectiveFunction(abc.ABC):
    """Abstract class for all objective functions."""
    @abc.abstractmethod
    def compute_loss(self, probs: Tensor, actions: Tensor, rewards: Tensor) -> Tensor:
        raise NotImplementedError


class AdvantageObjFunc(ObjectiveFunction):
    def compute_loss(self, probs, actions, rewards):
        mean_rewards = rewards.mean(dim=1).unsqueeze(1)
        deltas = rewards - mean_rewards
        action_probs = probs * actions + (1 - probs) * (1 - actions)
        losses = -deltas * action_probs
        loss = losses.sum(1).mean()
        return loss

class RewardObjFunc(ObjectiveFunction):
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

    def __repr__(self) -> str:
        return f"RewardObjectiveFunction(discount_factor={self.discount_factor})"


class EntropyObjFunc(ObjectiveFunction):
    def __init__(self, base_function: ObjectiveFunction, coefficient: float):
        self.base_function: ObjectiveFunction = base_function
        self.coefficient: float = coefficient

    def compute_loss(self, probs: Tensor, actions:Tensor, rewards: Tensor) -> Tensor:
        base_loss: Tensor = self.base_function.compute_loss(probs, actions, rewards)
        entropy_loss: Tensor = self._compute_entropy(probs)
        return base_loss - self.coefficient * entropy_loss

    def _compute_entropy(self, probs: Tensor) -> Tensor:
        two_class_probs = torch.stack([probs, 1 - probs], dim=2)
        entropy = -torch.sum(two_class_probs * torch.log2(two_class_probs + 1e-10), dim=2)
        return entropy.mean()

