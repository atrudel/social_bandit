import enum
from typing import List

import torch

class Choice(enum.IntEnum):
    PARTNER_1 = 0
    PARTNER_2 = 1

class History:
    def __init__(self):
        self.choices: List[Choice] = []
        self.points: List[int] = []

    def update(self, choice: Choice, points: int):
        self.choices.append(choice)
        self.points.append(points)

    def to_torch(self) -> torch.Tensor:
        choices = torch.tensor(self.choices).reshape(1, -1)
        points = torch.tensor(self.points).reshape(1, -1)
        return torch.stack([choices, points], dim=2)

    def __len__(self) -> int:
        return len(self.choices)

