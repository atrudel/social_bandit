import torch
from torchmetrics.classification import BinaryAccuracy

from social_bandit.config import N_TRIALS


def accuracy_metric(inputs, targets):
    acc_function = BinaryAccuracy()
    acc = acc_function(inputs, targets)
    return acc


def excess_reward_metric(actions, partner_trajectories, batch_average=True):
    not_chosen_actions = 1 - actions
    rewards = torch.gather(partner_trajectories, 1, actions.unsqueeze(1))
    missed_rewards = torch.gather(partner_trajectories, 1, not_chosen_actions.unsqueeze(1))
    excess_rewards = rewards - missed_rewards
    if batch_average:
        excess_rewards = excess_rewards.mean()
    return excess_rewards.mean(-1)


def imbalance_metric(actions):
    sum_partner_1 = actions.sum(1)
    imbalances = sum_partner_1 - N_TRIALS / 2
    return imbalances

def inequity_metric(actions, average=False):
    sum_partner_1 = actions.sum(1)
    imbalances = sum_partner_1 - N_TRIALS / 2
    inequitites = torch.abs(imbalances)
    if average:
        return inequitites.mean()
    else:
        return inequitites