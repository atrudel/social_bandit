import argparse

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


def quantitative_eval(model):
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

    print(f"Avg. Accuracy on test set: {acc:.3f}")
    print(f"Avg. Excess reward on test set: {mean_excess_rwd:.3f}")
    print(f"Avg. Imbalance on test set: {mean_imbalance:.3f}")

    plt.hist(excess_rwds.numpy(), bins=50)
    plt.axvline(mean_excess_rwd, c='red', label=f"Mean: {median_excess_rwd:.2f}")
    plt.axvline(median_excess_rwd, c='green', label=f"Median: {median_excess_rwd:.2f}")
    plt.title(f"Distribution of excess rewards on episodes of test set\n{model}")
    plt.xlim(xmin=-0.1, xmax=0.4)
    plt.xlabel(f"Excess reward (with rewards between 0 and 1)")
    plt.ylabel(f"Frequency out of {len(test_set)} test episodes")
    plt.legend()
    plt.show()

    plt.hist(imbalances.numpy(), bins=40)
    plt.axvline(mean_imbalance, c='red', label=f"Mean: {mean_imbalance:.2f}")
    plt.axvline(median_imbalance, c='green', label=f"Median: {median_imbalance:.2f}")
    plt.title(f"Distribution of imbalance on episodes of test set\n{model}")
    plt.xlim(xmin=-30, xmax=30)
    plt.xlabel("Action selection imbalance (Partner 1 - Partner 0)")
    plt.ylabel(f"Frequency out of {len(test_set)} test episodes")
    plt.legend()
    plt.show()

def uncertainty_generalization_eval(model, seed=None):
    def plot_heatmap(data):
        quantity: str = list(data)[-1]
        sns.heatmap(
            data=data.pivot(index='tau_samp', columns='tau_fluc', values=quantity).iloc[::-1],
            cmap='Blues',
            fmt=".2e"
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
            'tau_fluc': tau_fluc,
            'tau_samp': tau_samp,
            'accuracy': accuracy(actions, targets).item()
        })
        excess_rewards.loc[len(excess_rewards)] = pd.Series({
            'tau_fluc': tau_fluc,
            'tau_samp': tau_samp,
            'excess_reward': excess_reward(actions, trajectories).item()
        })
    plot_heatmap(accuracies)
    plt.title("Accuracies")
    plt.show()

    plot_heatmap(excess_rewards)
    plt.title("Excess Reward")
    plt.show()

def repeat_probability_eval(model):
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
                label=f'Probabilities by bins of {1/BINS:.2f} reward'
                )
    plt.title(f'Probability of repeating any action given the reward it received\n{model}')
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


if __name__ == '__main__':
    args = parser.parse_args()
    model = RNN.load(args.version)
    evaluate(model, seed=args.seed, quick=True)

