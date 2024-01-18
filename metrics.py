import torch
from torchmetrics.classification import BinaryAccuracy
import torch.nn as nn

def accuracy(actions, targets):
    return BinaryAccuracy()(actions, targets)

## Valid?
# def binary_crossentropy(actions, probs, targets):
#     return nn.functional.binary_crossentropy(probs, targets)

def excess_reward(actions, trajectories):
    not_chosen_actions = 1 - actions
    rewards = torch.gather(trajectories, 1, actions.unsqueeze(1))
    missed_rewards = torch.gather(trajectories, 1, not_chosen_actions.unsqueeze(1))
    excess_rewards = rewards - missed_rewards
    return excess_rewards.mean()