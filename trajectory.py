import abc

import numpy as np


class Trajectory(abc.ABC):
    def __init__(self, values: np.ndarray):
        self.values: np.ndarray = values

    def __getitem__(self, item):
        return self.values[item]

class ConstantTrajectory(Trajectory):
    def __init__(self, length: int, value: int):
        values = np.full(length, value)
        super().__init__(values)