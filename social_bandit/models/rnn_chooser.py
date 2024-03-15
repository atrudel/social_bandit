from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor


class RNNChooserPolicy(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int):
        super(RNNChooserPolicy, self).__init__()
        self.rnn = nn.RNN(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.hidden_state: Optional[Tensor] = None
        self._init_weights_xavier_uniform()

    def forward(self, last_action: Tensor, last_reward: Tensor) -> Tensor:
        rnn_input = torch.stack([last_action, last_reward], dim=2)
        out, hidden = self.rnn(rnn_input, self.hidden_state)
        out = out[:, -1, :]  # [batch, seq_len, hidden_size], taking the last hidden state of the sequence
        out = self.fc(out)
        out_prob = self.sigmoid(out)
        self.hidden_state = hidden
        return out_prob

    def reset(self, trajectory: Optional[Tensor] = None) -> None:
        # The RNN policy doesn't rely on a pre-determined trajectory
        self.hidden_state = None

    def _init_weights_xavier_uniform(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def __str__(self) -> str:
        return "RNNChooserPolicy"