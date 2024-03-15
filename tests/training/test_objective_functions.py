import math
from unittest.mock import Mock

import torch
from unittest.mock import MagicMock
from social_bandit.training.objective_functions import RewardObjFunc, EntropyObjFunc


def test_compute_returns_in_reward_obj_func():
    # Given
    rw_0 = 0.2
    rw_1 = 0
    rw_2 = 1
    rw_3 = 0.8
    rewards = torch.Tensor([rw_0, rw_1, rw_2, rw_3]).reshape(1, 4)

    discount = 0.5
    objective_function = RewardObjFunc(discount)

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
    objective_function = RewardObjFunc(discount)

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
    objective_function = RewardObjFunc(discount)

    # When
    loss = objective_function.compute_loss(probs, actions, rewards)

    # Then
    expected_loss = -((expected_returns[0] * math.log(0.1)).sum() +
                     (expected_returns[1] * math.log(1-0.6)).sum()) / 2
    torch.testing.assert_close(loss, expected_loss)

def test_compute_loss_in_reward_obj_func_single_example():
    # Given
    rewards = torch.tensor([[0.1, 0.6, 0.8, 0]])
    probs = torch.tensor([[0.4, 0.5, 0.9, 0.3]])
    actions = torch.tensor([[0, 1, 1, 0]])

    objective_function = RewardObjFunc(discount_factor=0.5)


    # When
    returns = objective_function._compute_returns(rewards)
    loss = objective_function.compute_loss(probs, actions, rewards)

    # Then
    expected_returns = torch.tensor([[0.6, 1.0, 0.8, 0]])
    torch.testing.assert_close(returns, expected_returns)

    expected_loss = -(
        math.log(1 - probs[0][0]) * returns[0][0] +   # Action 0
        math.log(probs[0][1]) * returns[0][1] +         # Action 1
        math.log(probs[0][2]) * returns[0][2] +         # Action 1
        math.log(1 - probs[0][3]) * returns[0][3]     # Action 0
    )
    torch.testing.assert_close(loss, expected_loss)

def test_entropy_loss_with_uniform_distribution_maximum_entropy():
    # Given
    probs = torch.full((2, 5), fill_value=0.5)
    mock_base_loss = torch.tensor(3.5)

    mock_base_function = Mock()
    mock_base_function.compute_loss = Mock(return_value=mock_base_loss)
    coef = 0.3
    obj_func = EntropyObjFunc(mock_base_function, coefficient=coef)

    # When
    entropy = obj_func._compute_entropy(probs)
    loss = obj_func.compute_loss(probs, None, None)

    # Then
    expected_entropy = torch.tensor(1, dtype=torch.float)
    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(loss, mock_base_loss - coef * expected_entropy)


def test_entropy_loss_with_random_distribution():
    # Given
    probs = torch.tensor([[0.1, 0.6, 0.5],
                        [0.9, 0.8, 0.2,]])
    mock_base_loss = torch.tensor(3.5)

    mock_base_function = Mock()
    mock_base_function.compute_loss = Mock(return_value=mock_base_loss)
    coef = 0.3
    obj_func = EntropyObjFunc(mock_base_function, coefficient=coef)

    # When
    entropy = obj_func._compute_entropy(probs)
    loss = obj_func.compute_loss(probs, None, None)

    # Then
    expected_entropy = torch.tensor([
        - 0.1 * math.log2(0.1) - (1 - 0.1) * math.log2(1 - 0.1),
        - 0.6 * math.log2(0.6) - (1 - 0.6) * math.log2(1 - 0.6),
        - 0.5 * math.log2(0.5) - (1 - 0.5) * math.log2(1 - 0.5),
        - 0.9 * math.log2(0.9) - (1 - 0.9) * math.log2(1 - 0.9),
        - 0.8 * math.log2(0.8) - (1 - 0.8) * math.log2(1 - 0.8),
        - 0.2 * math.log2(0.2) - (1 - 0.2) * math.log2(1 - 0.2)
    ]).mean()

    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(loss, mock_base_loss - coef * expected_entropy)

def test_entropy_loss_behaves_like_base_function_when_coefficient_is_0():
    # Given
    rewards = torch.tensor([[0.1, 0.6, 0.8, 0]])
    probs = torch.tensor([[0.4, 0.5, 0.9, 0.3]])
    actions = torch.tensor([[0, 1, 1, 0]])

    base_objective_function = RewardObjFunc(discount_factor=0.5)
    entropy_objective_function = EntropyObjFunc(base_objective_function, coefficient=0)

    # When
    loss = entropy_objective_function.compute_loss(probs, actions, rewards)

    # Then
    expected_loss = base_objective_function.compute_loss(probs, actions, rewards)
    torch.testing.assert_close(loss, expected_loss)