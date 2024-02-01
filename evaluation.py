import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader
import seaborn as sns
from tqdm import tqdm

from RNN import RNN
from config import DEVICE, POINTS_PER_TURN, SEQUENCE_LENGTH, DATA_DIR
from data_generator import BanditGenerator, BanditDataset
from metrics import accuracy, excess_reward

parser = argparse.ArgumentParser(description="Evaluation of model.")
parser.add_argument('--model_dir', type=str, required=True, help='Directory where the model checkpoint is located.')
parser.add_argument('--seed',  type=int, default=42, help='Random state to use for reproducibility')


def quantitative_eval(model):
    model.eval()
    test_set = BanditDataset('test', DATA_DIR)  # Todo: handle this better
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))
    batch = list(test_dataloader)[0]

    actions, probs, rewards, targets, trajectories = model.process_trajectory(batch)
    acc = accuracy(actions, targets).item()
    excess_rwds = excess_reward(actions, trajectories, batch_average=False)

    print("Avg. Accuracy on test set: ", acc)
    print("Avg. Excess reward on test set: ", excess_rwds.mean().item())

    plt.hist(excess_rwds.numpy(), bins=50)
    plt.title("Distribution of excess rewards on episodes of test set")
    plt.show()

def uncertainty_generalization_eval(model, seed=None, bins=8, batch_size=100):
    def plot_heatmap(data):
        quantity: str = list(data)[-1]
        sns.heatmap(
            data=data.pivot(index='tau_samp', columns='tau_fluc', values=quantity).iloc[::-1],
            cmap='Blues',
            fmt=".2e"
        )

    model.eval()
    tau_flucs = np.linspace(1, 5, num=bins)
    tau_samps = np.linspace(0, 4, num=bins)
    accuracies = pd.DataFrame(columns=['tau_fluc', 'tau_samp', 'accuracy'])
    excess_rewards = pd.DataFrame(columns=['tau_fluc', 'tau_samp', 'excess_reward'])

    progress_bar = tqdm(total=bins**2, desc=f'Generating evaluation sets of size {batch_size}')
    for tau_fluc in tau_flucs:
        for tau_samp in tau_samps:
            data_generator = BanditGenerator(tau_fluc=tau_fluc, tau_samp=tau_samp, seed=seed, verbose=False)
            means, values = data_generator.generate_batch(batch_size=batch_size, length=SEQUENCE_LENGTH)
            dataset = BanditDataset(values=values, means=means)
            dataloader = DataLoader(dataset, batch_size=batch_size)
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
            progress_bar.update(1)
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
    test_set = BanditDataset('test', DATA_DIR)  # Todo: handle this better
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))
    batch = list(test_dataloader)[0]

    # Produce model behavior
    actions, probs, rewards, targets, trajectories = model.process_trajectory(batch)
    next_actions = actions.roll(-1, dims=1)
    repeats = (actions == next_actions).float()[:, :-1]

    # Compute statistics
    rpts = repeats.flatten().numpy()
    rwds = rewards[:, :-1].flatten().numpy() * POINTS_PER_TURN
    data = pd.DataFrame({
        'reward': rwds,
        'repeat': rpts
    })
    groups = data.groupby(pd.cut(rwds, bins=BINS), observed=True)
    probs_by_bin = groups.mean()

    # Plot statistics by bins
    plt.scatter(probs_by_bin['reward'], probs_by_bin['repeat'],
                label=f'Probabilities by bins of {POINTS_PER_TURN/BINS} pts of reward'
                )
    plt.title('Probability of repeating any action given the reward it received')
    plt.ylabel('Probability of repeating last action')
    plt.xlabel('Reward of last action')

    # Fit a logistic function
    def logistic_function(x, x0, k):
        return 1 / (1 + np.exp(-k * (x - x0)))

    params, _ = curve_fit(logistic_function,
                           probs_by_bin['reward'].values.astype(np.float64),
                           probs_by_bin['repeat'].values.astype(np.float64),
                           p0=[POINTS_PER_TURN / 2, 1])
    x_curve = np.linspace(0, POINTS_PER_TURN, num=100)
    y_curve = logistic_function(x_curve, *params)
    plt.plot(x_curve, y_curve,
             label=f"Fitted curve: f(x) = 1 / (1 + exp(-{params[1]:.2f} * (x - {params[0]:.2f}))"
             )
    plt.legend()
    plt.show()

def evaluate(model: RNN, seed):
    model.eval()
    quantitative_eval(model)
    uncertainty_generalization_eval(model, seed)
    repeat_probability_eval(model)


if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint_path = glob.glob(os.path.join(
        args.model_dir,
        'checkpoints/*.ckpt'
    ))[-1]
    print(f"Evaluating model with checkpoint {checkpoint_path}")

    model = RNN.load_from_checkpoint(checkpoint_path, map_location=DEVICE)

    evaluate(model, seed=args.seed)

