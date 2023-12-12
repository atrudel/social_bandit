import numpy as np

from config import POINTS_PER_TURN
from data import Trajectory


class Partner:
    def __init__(self, trajectory: Trajectory):
        self.trajectory: Trajectory = trajectory
        self.history = []
        self.score = 0

    def play_turn(self, turn: int) -> int:
        returned_points: int = self.trajectory[turn]
        kept_points: int = POINTS_PER_TURN - returned_points
        self.history.append(kept_points)
        self.score += kept_points
        return returned_points

    def pass_turn(self, turn: int):
        self.history.append(0)

