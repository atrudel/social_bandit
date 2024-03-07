from abc import abstractmethod
from typing import Optional, Callable

from torch import Tensor

from bandit_game.utils import History


class Partner:
    def __init__(self):
        self.history: History = History()

    def play_trial(self) -> Tensor:
        if len(self.history) == 0:
            return self.first_trial_policy()
        else:
            last_action, _ = self.history.get_last_trial()
            return self.policy(last_action)

    def update_record(self, choice: Tensor, reward: Tensor):
        self.history.update(choice, reward)

    @abstractmethod
    def first_trial_policy(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def policy(self, last_action: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def reset(self, trajectory: Optional[Tensor] = None) -> None:
        raise NotImplementedError


class DataPartner(Partner):
    def __init__(self):
        super().__init__()
        self.trajectory: Optional[Tensor] = None

    def policy(self, last_action: Tensor) -> Tensor:
        assert self.trajectory is not None, "DataPartner must have trajectory data loaded before it can choose actions"
        return self.trajectory[:, len(self.history)].reshape(-1, 1)

    def first_trial_policy(self) -> Tensor:
        return self.policy(None)

    def reset(self, trajectory: Optional[Tensor] = None) -> None:
        self.history = History()
        self.trajectory = trajectory

    def load_trajectory(self, trajectory: Tensor):
        assert trajectory.ndim == 2, "Trajectory array should have 2 dimensions: batch x length"
        self.trajectory = trajectory
