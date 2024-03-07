from typing import List, Tuple, Optional

import torch
from torch import Tensor


class History:
    def __init__(self):
        self.choices: List[Tensor] = []
        self.points: List[Tensor] = []
        self.probs: List[Tensor] = []

    def update(self, choice: Tensor, points: Tensor, prob: Optional[Tensor] = None) -> None:
        self.choices.append(choice)
        self.points.append(points)
        if prob is not None:
            self.probs.append(prob)

    def add_choice_and_prob(self, choice: Tensor, prob: Tensor):
        self.choices.append(choice)
        self.probs.append(prob)

    def add_points(self, points: Tensor):
        self.points.append(points)
        self._check_data_integrity()

    def get_last_trial(self) -> Tuple[Tensor, Tensor]:
        self._check_data_integrity()
        return self.choices[-1], self.points[-1]

    def get_full_trajectory(self):
        probs = torch.cat(self.probs, dim=1)
        choices = torch.cat(self.choices, dim=1)
        points = torch.cat(self.points, dim=1)
        return probs, choices, points

    def to_torch(self) -> torch.Tensor:
        choices = torch.tensor(self.choices).reshape(1, -1)
        points = torch.tensor(self.points).reshape(1, -1)
        return torch.stack([choices, points], dim=2)

    def _check_data_integrity(self):
        assert len(self.choices) == len(self.points), "Number of Choice and Point entries do not match"
        if len(self.probs) > 0:
            assert len(self.probs) == len(self.choices), "Number of Probs and Choice entries do not match"

    def __len__(self) -> int:
        self._check_data_integrity()
        return len(self.choices)

