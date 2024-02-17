import argparse
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader

from RNN import RNN
from config import POINTS_PER_TURN, DATA_DIR
from data_generator import GeneralizationDatasetBundle
from dataset import BanditDataset
from metrics import accuracy, excess_reward, imbalance

parser = argparse.ArgumentParser(description="Evaluation of model.")
parser.add_argument('--version', type=str, required=True, help='Version name of the model to load')
parser.add_argument('--seed',  type=int, default=1, help='Random state to use for reproducibility')

HIST_HEIGHT = 3
HIST_WIDTH = 4

def quantitative_eval(model, show=True, axes=None):
    model.eval()
    test_set = BanditDataset.load(name='test', directory=DATA_DIR)
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))
    batch = list(test_dataloader)[0]

    actions, probs, rewards, targets, trajectories = model.process_trajectory(batch)
    acc = accuracy(actions, targets).item()
    excess_rwds = excess_reward(actions, trajectories, batch_average=False)
    imbalances = imbalance(actions)

    mean_excess_rwd = excess_rwds.mean().item()
    median_excess_rwd = excess_rwds.median().item()
    mean_imbalance = imbalances.mean().item()
    median_imbalance = imbalances.median().item()
    print(model)
    print('-' * len(str(model)))
    print(f"Avg. Accuracy on test set: {acc:.3f}")
    print(f"Avg. Excess reward on test set: {mean_excess_rwd:.3f}")
    print(f"Avg. Imbalance on test set: {mean_imbalance:.3f}")

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot excess rewards histogram
    axes[0].hist(excess_rwds.numpy(), bins=50)
    axes[0].axvline(mean_excess_rwd, c='red', label=f"Mean: {median_excess_rwd:.2f}")
    axes[0].axvline(median_excess_rwd, c='green', label=f"Median: {median_excess_rwd:.2f}")
    axes[0].set_title(f"Distribution of excess rewards")
    axes[0].set_xlim(xmin=-0.1, xmax=0.4)
    axes[0].set_xlabel(f"Excess reward")
    axes[0].set_ylabel(f"Frequency (/{len(test_set)} episodes")
    axes[0].legend()

    # Plot imbalances histogram
    axes[1].hist(imbalances.numpy(), bins=40)
    axes[1].axvline(mean_imbalance, c='red', label=f"Mean: {mean_imbalance:.2f}")
    axes[1].axvline(median_imbalance, c='green', label=f"Median: {median_imbalance:.2f}")
    axes[1].set_title(f"Distribution of imbalance")
    axes[1].set_xlim(xmin=-30, xmax=30)
    axes[1].set_xlabel("Action selection imbalance (Partner 1 - Partner 0)")
    axes[1].legend()

    if show:
        plt.tight_layout()
        plt.show()

    data =  pd.DataFrame({
        'Reward_loss': [model.reward_loss_coef],
        'Equity_loss': [model.equity_loss_coef],
        'Accuracy': [acc],
        'Excess reward': [mean_excess_rwd],
        'Imbalance': [mean_imbalance]
    })
    data = data.set_index(['Reward_loss', 'Equity_loss'])

    return data


def uncertainty_generalization_eval(model, seed=None):
    def plot_heatmap(data, ax):
        quantity: str = list(data)[-1]
        sns.heatmap(
            data=data.pivot(index='tau_samp', columns='tau_fluc', values=quantity).iloc[::-1],
            fmt='.2f',
            ax=ax,
            cmap='Blues'
        )

    model.eval()
    np.random.seed(seed)
    accuracies = pd.DataFrame(columns=['tau_fluc', 'tau_samp', 'accuracy'])
    excess_rewards = pd.DataFrame(columns=['tau_fluc', 'tau_samp', 'excess_reward'])

    datasets = GeneralizationDatasetBundle.load(DATA_DIR)
    for tau_fluc, tau_samp, dataset in datasets:
        dataset = datasets.get(tau_fluc, tau_samp)
        dataloader = DataLoader(dataset, batch_size=len(dataset))
        batch = list(dataloader)[0]

        with torch.no_grad():
            actions, probs, rewards, targets, trajectories = model.process_trajectory(batch)

        accuracies.loc[len(accuracies)] = pd.Series({
            'tau_fluc': np.round(tau_fluc, 2),
            'tau_samp': np.round(tau_samp, 2),
            'accuracy': accuracy(actions, targets).item()
        })
        excess_rewards.loc[len(excess_rewards)] = pd.Series({
            'tau_fluc': np.round(tau_fluc, 2),
            'tau_samp': np.round(tau_samp, 2),
            'excess_reward': excess_reward(actions, trajectories).item()
        })

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    plot_heatmap(accuracies, axs[0])
    axs[0].set_title("Accuracies")

    plot_heatmap(excess_rewards, axs[1])
    axs[1].set_title("Excess Reward")
    plt.tight_layout()
    plt.show()

def repeat_probability_eval(model, show=True):
    BINS = 20

    model.eval()
    # Load test dataset
    test_set = BanditDataset.load(name='test', directory=DATA_DIR)
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))
    batch = list(test_dataloader)[0]

    # Produce model behavior
    actions, probs, rewards, targets, trajectories = model.process_trajectory(batch)
    next_actions = actions.roll(-1, dims=1)
    repeats = (actions == next_actions).float()[:, :-1]

    # Compute statistics
    rpts = repeats.flatten().numpy()
    rwds = rewards[:, :-1].flatten().numpy()
    data = pd.DataFrame({
        'reward': rwds,
        'repeat': rpts
    })
    groups = data.groupby(pd.cut(rwds, bins=BINS), observed=True)
    probs_by_bin = groups.mean()

    # Plot statistics by bins
    plt.scatter(probs_by_bin['reward'], probs_by_bin['repeat'],
                label=str(model)
                )
    plt.ylim(0, 1)
    plt.title(f'Probability of repeating any action given the reward it received (bins of {1/BINS:.2f})')
    plt.ylabel('Probability of repeating last action')
    plt.xlabel('Reward of last action')

    # Fit a logistic function
    def logistic_function(x, x0, a, baseline):
        return baseline + (1 - baseline) / (1 + np.exp(-a * (x - x0)))

    try:
        params, _ = curve_fit(logistic_function,
                               probs_by_bin['reward'].values.astype(np.float64),
                               probs_by_bin['repeat'].values.astype(np.float64),
                              p0=[0.5, 5, 0.3])
        x_curve = np.linspace(0, 1, num=100)
        y_curve = logistic_function(x_curve, *params)
        plt.plot(x_curve, y_curve,
                 label=f"Fitted curve: f(x) = {params[2]:.2f} + {1-params[2]:.2f} / (1 + exp(-{params[1]:.2f}( x - {params[0]:.2f})))"
                 )
    except RuntimeError:
        print("Unable to fit sigmoid curve")
    plt.legend()
    if show:
        plt.show()

def visualize_play(model: RNN, idx: int = 0):
    model.eval()
    test_set = BanditDataset.load(name='test', directory=DATA_DIR)
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))
    batch = list(test_dataloader)[0]

    actions, probs, rewards, targets, trajectories = model.process_trajectory(batch)
    actions_to_plot = actions[idx]
    probs_to_plot = probs[idx].detach().numpy()
    test_set.plot(idx, show=False)
    plt.scatter(list(range(len(actions_to_plot))), actions_to_plot + 0.05,
                label='Model actions', marker='+', c='red')
    plt.plot(list(range(len(probs_to_plot))), probs_to_plot, c='red', label='Model output probability')
    plt.legend(bbox_to_anchor=(1, 0.5), loc="upper left")
    plt.show()

def evaluate(model: RNN, seed):
    torch.manual_seed(seed)
    model.eval()
    quantitative_eval(model)
    uncertainty_generalization_eval(model, seed)
    repeat_probability_eval(model)
    visualize_play(model)


def comparative_evaluation(versions: List[str], seed: int) -> pd.DataFrame:
    models = [RNN.load(version) for version in versions]
    stats = []

    # Quantitative evaluation
    fig, axes = plt.subplots(len(models), 2, figsize=(2 * HIST_WIDTH, HIST_HEIGHT * len(models)), sharey=True, sharex=True)
    for i, model in enumerate(models):
        torch.manual_seed(seed)
        model_stats = quantitative_eval(model, show=False, axes=axes[i])
        axes[i][1].text(-55, 50, model.multiline_str(),
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
        stats.append(model_stats)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig('plot')
    plt.show()

    # Repeat Probability evaluation
    for model in models:
        torch.manual_seed(seed)
        repeat_probability_eval(model, show=False)
    plt.show()

    return pd.concat(stats)


if __name__ == '__main__':
    args = parser.parse_args()
    model = RNN.load('version_25')
    # repeat_probability_eval(model)
    comparative_evaluation(['version_24', 'version_25', 'version_26', 'version_27'], 42)


