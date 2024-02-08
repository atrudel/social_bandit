import torch
from torchmetrics.classification import BinaryAccuracy
import torch.nn as nn

from config import SEQUENCE_LENGTH


def accuracy(inputs, targets):
    return BinaryAccuracy()(inputs, targets)


def excess_reward(actions, trajectories, batch_average=True):
    not_chosen_actions = 1 - actions
    rewards = torch.gather(trajectories, 1, actions.unsqueeze(1))
    missed_rewards = torch.gather(trajectories, 1, not_chosen_actions.unsqueeze(1))
    excess_rewards = rewards - missed_rewards
    if batch_average:
        excess_rewards = excess_rewards.mean()
    return excess_rewards.mean(-1)

def inequity(actions):
    sum_partner_1 = actions.sum(1)
    inequities = sum_partner_1 - SEQUENCE_LENGTH / 2
    return inequities