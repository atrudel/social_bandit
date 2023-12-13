from typing import Union

import numpy as np


class Partner:
    def __init__(self, trajectories: np.ndarray):
        self.trajectories: np.ndarray = trajectories
        assert self.trajectories.ndim == 2, "Trajectory array should have 2 dimensions: batch x length"

    def play_trial(self, trial: int, traj_idx: int = None) -> Union[int, np.ndarray]:
        # Play on a batch of games
        if traj_idx is None:
            return self.trajectories[:, trial]
        # Play only one game, specified by traj_idx
        else:
            return self.trajectories[traj_idx, trial]

    def __len__(self):
        return self.trajectories.shape[1]

