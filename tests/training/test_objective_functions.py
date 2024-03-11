import math

import torch

from social_bandit.training import RewardObjectiveFunction


def test_compute_returns_in_reward_obj_func():
    # Given
    rw_0 = 0.2
    rw_1 = 0
    rw_2 = 1
    rw_3 = 0.8
    rewards = torch.Tensor([rw_0, rw_1, rw_2, rw_3]).reshape(1, 4)

    discount = 0.5
    objective_function = RewardObjectiveFunction(discount)

    # When
    returns = objective_function._compute_returns(rewards)

    # Then
    assert returns[0, 3] == rw_3
    assert returns[0, 2] == rw_2 + rw_3 * discount
    assert returns[0, 1] == rw_1 + rw_2 * discount + rw_3 * discount**2
    assert returns[0, 0] == rw_0 + rw_1 * discount + rw_2 * discount**2 + rw_3 * discount**3


# Fixtures
rewards = torch.Tensor([[0.2, 0, 1, 0.8],
                        [0.1, 0.2, 0, 0.8]])

expected_returns = torch.Tensor([[0.55, 0.7, 1.4, 0.8],
                                 [0.3, 0.4, 0.4, 0.8]])
def test_compute_returns_in_reward_obj_func_for_2d_tensor():
    # Given
    # reward_tensor and returns_tensor declared above
    discount = 0.5
    objective_function = RewardObjectiveFunction(discount)

    # When
    returns = objective_function._compute_returns(rewards)

    torch.testing.assert_close(returns, expected_returns)


def test_compute_loss_in_reward_obj_func():
    # Given
    ## reward_tensor declared above
    discount = 0.5
    probs = torch.tensor([[0.1, 0.1, 0.1, 0.1],
                          [0.6, 0.6, 0.6, 0.6]])
    actions = torch.tensor([[1, 1, 1, 1],
                            [0, 0, 0, 0]])
    objective_function = RewardObjectiveFunction(discount)

    # When
    loss = objective_function.compute_loss(probs, actions, rewards)

    # Then
    expected_loss = -((expected_returns[0] * math.log(0.1)).sum() +
                     (expected_returns[1] * math.log(1-0.6)).sum()) / 2
    torch.testing.assert_close(loss, expected_loss)
