from __future__ import annotations
from pathlib import Path
from typing import Union

import torch
from torch import nn

from config import DEVICE
from training.objective_functions import ObjectiveFunction


class RNNforBinaryAction(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super(RNNforBinaryAction, self).__init__()
        self.rnn = nn.RNN(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, h_0):
        out, hidden = self.rnn(input, h_0)
        out = out[:, -1, :]  # [batch, seq_len, hidden_size], taking the last hidden state of the sequence
        out = self.fc(out)
        out = self.sigmoid(out)
        return out, hidden

    # @classmethod
    # def load(cls, save_path: Union[str, Path]) -> RNNChooser:
    #     checkpoint = torch.load(save_path, map_location=DEVICE)
    #     model_state_dict = checkpoint['model_state_dict']
    #     model = RNNChooser(1,1)
    #     model.load_state_dict(model_state_dict)
    #     model.eval()
    #     return model

