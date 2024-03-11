import argparse
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader

from social_bandit.models import RNN
from social_bandit.config import DATA_DIR
from social_bandit.data_generation.data_generator import GeneralizationDatasetBundle
from social_bandit.data_generation.dataset import BanditDataset
from metrics import accuracy_metric, excess_reward_metric, imbalance_metric

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
    acc = accuracy_metric(actions, targets).item()
    excess_rwds = excess_reward_metric(actions, trajectories, batch_average=False)
    imbalances = imbalance_metric(actions)

    mean_excess_rwd = excess_rwds.mean().item()
    median_excess_rwd = excess_rwds.median().item()
    mean_imbalance = imbalances.mean().item()
    median_imbalance = imbalances.median().item()

    if show:
        print(model)
        print('-' * len(str(model)))
        print(f"Avg. Accuracy on test set: {acc:.3f}")
        print(f"Avg. Excess reward on test set: {mean_excess_rwd:.3f}")
        print(f"Avg. Imbalance on test set: {mean_imbalance:.3f}")

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot excess rewards histogram
    ax_rwd = axes[0]
    ax_rwd.hist(excess_rwds.numpy(), bins=50)
    ax_rwd.axvline(mean_excess_rwd, c='red', label=f"Mean: {median_excess_rwd:.2f}")
    ax_rwd.axvline(median_excess_rwd, c='green', label=f"Median: {median_excess_rwd:.2f}")
    ax_rwd.set_title(f"Distribution of excess rewards")
    ax_rwd.set_xlim(xmin=-0.1, xmax=0.4)
    ax_rwd.set_xlabel(f"Excess reward")
    ax_rwd.set_ylabel(f"Frequency (/{len(test_set)} episodes")
    ax_rwd.legend()

    # Plot imbalances histogram
    ax_imbalance = axes[1]
    ax_imbalance.hist(imbalances.numpy(), bins=40)
    ax_imbalance.axvline(mean_imbalance, c='red', label=f"Mean: {mean_imbalance:.2f}")
    ax_imbalance.axvline(median_imbalance, c='green', label=f"Median: {median_imbalance:.2f}")
    ax_imbalance.set_title(f"Distribution of imbalance")
    ax_imbalance.set_xlim(xmin=-30, xmax=30)
    ax_imbalance.set_xlabel("Action selection imbalance (Partner 1 - Partner 0)")
    ax_imbalance.legend()

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
            'accuracy': accuracy_metric(actions, targets).item()
        })
        excess_rewards.loc[len(excess_rewards)] = pd.Series({
            'tau_fluc': np.round(tau_fluc, 2),
            'tau_samp': np.round(tau_samp, 2),
            'excess_reward': excess_reward_metric(actions, trajectories).item()
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
    scatter = plt.scatter(probs_by_bin['reward'], probs_by_bin['repeat'], label=str(model))
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
                 label=f"Fitted curve: f(x) = {params[2]:.2f} + {1-params[2]:.2f} / (1 + exp(-{params[1]:.2f}( x - {params[0]:.2f})))",
                 c=scatter.get_facecolor()[0]
                 )
    except RuntimeError:
        print("Unable to fit sigmoid curve")
    plt.legend()
    if show:
        plt.show()

def visualize_play(model: RNN, idx: int = 0, show: bool = True, color=None):
    model.eval()
    test_set = BanditDataset.load(name='test', directory=DATA_DIR)
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))
    batch = list(test_dataloader)[0]

    actions, probs, rewards, targets, trajectories = model.process_trajectory(batch)
    actions_to_plot = actions[idx]
    probs_to_plot = probs[idx].detach().numpy()
    if color is None:
        test_set.plot(idx, show=False)
        plt.scatter(list(range(len(actions_to_plot))), actions_to_plot + 0.05,
                label='Model actions', marker='+', c='red')
        plt.plot(list(range(len(probs_to_plot))), probs_to_plot, c='red', label=f"Output prob {model}")
    else:
        plt.plot(list(range(len(probs_to_plot))), probs_to_plot, c=color, label=f"{model}")
    plt.ylabel('Bandit reward / Model Output Probability')
    plt.legend(bbox_to_anchor=(1, 0.5), loc="upper left")
    if show:
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

    # Repeat Probability evaluation
    for model in models:
        torch.manual_seed(seed)
        repeat_probability_eval(model, show=False)
    plt.legend(bbox_to_anchor=(1, 0.5), loc="upper left")
    plt.show()

    # Visualize play
    test_set = BanditDataset.load(name='test', directory=DATA_DIR)
    test_set.plot(0, show=False, simplified=True)
    color_cycle = ['Green', 'Red', 'Gray', 'Brown', 'Pink', 'Purple']
    for i, model in enumerate(models):
        visualize_play(model, 0, show=False, color=color_cycle[i])
    plt.show()

    # Quantitative evaluation
    fig, axes = plt.subplots(len(models), 2, figsize=(2 * HIST_WIDTH, HIST_HEIGHT * len(models)))
    stats = []
    for i, model in enumerate(models):
        torch.manual_seed(seed)
        model_stats = quantitative_eval(model, show=False, axes=axes[i])
        axes[i][1].text(-55, 50, model.multiline_str(),
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
        stats.append(model_stats)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig('plot')
    plt.show()
    return pd.concat(stats)


if __name__ == '__main__':
    args = parser.parse_args()
    # quantitative_eval(RNN.load('version_36'))
    # quantitative_eval(RNN.load('version_37'))
    comparative_evaluation(['version_35', 'version_36', 'version_37'], 42)


