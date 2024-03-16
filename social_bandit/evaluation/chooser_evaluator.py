from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from torch import nn, Tensor
from torch.utils.data import DataLoader

from social_bandit.config import DATA_DIR
from social_bandit.data_generation.data_generator import GeneralizationDatasetBundle
from social_bandit.data_generation.dataset import BanditDataset
from social_bandit.evaluation.metrics import accuracy_metric, excess_reward_metric, imbalance_metric
from social_bandit.game.environment import Env

parser = argparse.ArgumentParser(description="Evaluation of model.")
parser.add_argument('--version', type=str, required=True, help='Version name of the model to load')
parser.add_argument('--seed',  type=int, default=1, help='Random state to use for reproducibility')

HIST_HEIGHT = 3
HIST_WIDTH = 4


class ChooserEvaluator:
    def __init__(self, envs: List[Env], save_dir: Path = None):
        self.envs: List[Env] = envs
        self.save_dir: Path = save_dir

    def run_evaluations(self, seed=None, show=True):
        torch.manual_seed(seed)
        partners_trajectories: Tensor = self._load_test_data()
        test_trajectories: List[Dict[str, Tensor]] = self._generate_test_trajectories(self.envs, partners_trajectories)
        quantitative_results = self.quantitative_evaluation(test_trajectories, show=show)
        self.repeat_probability_eval(test_trajectories, show=show)
        self.uncertainty_generalization_eval(seed=seed, show=show)
        self.trajectory_visualization(test_idx=0, show=show)
        return quantitative_results

    def quantitative_evaluation(self, trajectories: List[dict], show=True):

        results = pd.DataFrame(columns=['Agent descr.',
                                        'Accuracy (avg)',
                                        'Excess reward (avg)',
                                        'Excess reward (med)',
                                        'Imbalance (avg)',
                                        'Imbalance (med)']).set_index('Agent descr.')

        fig, axes = plt.subplots(len(trajectories), 2, figsize=(2 * HIST_WIDTH, HIST_HEIGHT * len(trajectories)))
        axes = axes.reshape(-1, 2)

        for i, trajectory in enumerate(trajectories):
            mean_accuracy = accuracy_metric(trajectory['actions'], trajectory['targets']).item()
            excess_rewards = excess_reward_metric(trajectory['actions'], trajectory['partner_trajectories'], batch_average=False)
            imbalances = imbalance_metric(trajectory['actions'])

            mean_excess_reward = excess_rewards.mean().item()
            median_excess_reward = excess_rewards.median().item()
            mean_imbalance = imbalances.mean().item()
            median_imbalance = imbalances.median().item()

            new_data = {
                'Accuracy (avg)': mean_accuracy,
                'Excess reward (avg)': mean_excess_reward,
                'Excess reward (med)': median_excess_reward,
                'Imbalance (avg)': mean_imbalance,
                'Imbalance (med)': median_imbalance
            }
            results.loc[trajectory['description']] = new_data

            # Plot excess rewards histogram
            ax_rwd = axes[i][0]
            ax_rwd.hist(excess_rewards.numpy(), bins=50, label=trajectory['description'])
            ax_rwd.axvline(mean_excess_reward, c='red', label=f"Mean: {median_excess_reward:.2f}")
            ax_rwd.axvline(median_excess_reward, c='green', label=f"Median: {median_excess_reward:.2f}")
            ax_rwd.set_title(f"Distribution of excess rewards")
            ax_rwd.set_xlim(xmin=-0.1, xmax=0.4)
            ax_rwd.set_xlabel(f"Excess reward")
            ax_rwd.set_ylabel(f"Frequency (/{trajectory['actions'].shape[0]} episodes")
            ax_rwd.legend()

            # Plot imbalances histogram
            ax_imbalance = axes[i][1]
            ax_imbalance.hist(imbalances.numpy(), bins=40, label=trajectory['description'])
            ax_imbalance.axvline(mean_imbalance, c='red', label=f"Mean: {mean_imbalance:.2f}")
            ax_imbalance.axvline(median_imbalance, c='green', label=f"Median: {median_imbalance:.2f}")
            ax_imbalance.set_title(f"Distribution of imbalance")
            ax_imbalance.set_xlim(xmin=-30, xmax=30)
            ax_imbalance.set_xlabel("Action selection imbalance (Partner 1 - Partner 0)")
            ax_imbalance.legend()

        plt.legend(loc='best')
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.tight_layout()

        if self.save_dir is not None:
            plt.savefig(self.save_dir / 'quantitative_plots.png')
            results.to_csv(self.save_dir / 'quantitative_results.csv')

        if show:
            plt.show()
        else:
            plt.close()
        return results

    def repeat_probability_eval(self, trajectories: List[dict], show=True):
        BINS = 20

        def logistic_function(x, x0, a, baseline):
            return baseline + (1 - baseline) / (1 + np.exp(-a * (x - x0)))

        for trajectory in trajectories:
            # Compute repetitions of actions
            rewards = trajectory['rewards']
            actions = trajectory['actions']
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
            scatter = plt.scatter(probs_by_bin['reward'], probs_by_bin['repeat'], label=trajectory['description'])
            plt.ylim(0, 1)
            plt.title(f'Probability of repeating any action given the reward it received (bins of {1 / BINS:.2f})')
            plt.ylabel('Probability of repeating last action')
            plt.xlabel('Reward of last action')

        try:
            params, _ = curve_fit(logistic_function,
                                  probs_by_bin['reward'].values.astype(np.float64),
                                  probs_by_bin['repeat'].values.astype(np.float64),
                                  p0=[0.5, 5, 0.3])
            x_curve = np.linspace(0, 1, num=100)
            y_curve = logistic_function(x_curve, *params)
            plt.plot(x_curve, y_curve,
                     label=f"Fitted curve: f(x) = {params[2]:.2f} + {1 - params[2]:.2f} / (1 + exp(-{params[1]:.2f}( x - {params[0]:.2f})))",
                     c=scatter.get_facecolor()[0]
                     )
        except RuntimeError:
            print("Unable to fit sigmoid curve")
        plt.legend()

        if self.save_dir is not None:
            plt.savefig(self.save_dir / 'repeat_probability.png')

        if show:
            plt.show()
        else:
            plt.close()

    def uncertainty_generalization_eval(self, seed=None, show=True):
        def plot_heatmap(data, ax):
            quantity: str = list(data)[-1]
            sns.heatmap(
                data=data.pivot(index='tau_samp', columns='tau_fluc', values=quantity).iloc[::-1],
                fmt='.2f',
                ax=ax,
                cmap='Blues'
            )

        fig, axs = plt.subplots(len(self.envs), 2, sharey=True)
        axs = axs.reshape(-1, 2)
        datasets = GeneralizationDatasetBundle.load(DATA_DIR)

        for i, env in enumerate(self.envs):
            env.chooser.policy.eval()
            np.random.seed(seed)
            accuracies = pd.DataFrame(columns=['tau_fluc', 'tau_samp', 'accuracy'])
            excess_rewards = pd.DataFrame(columns=['tau_fluc', 'tau_samp', 'excess_reward'])

            for tau_fluc, tau_samp, dataset in datasets:
                dataset = datasets.get(tau_fluc, tau_samp)
                dataloader = DataLoader(dataset, batch_size=len(dataset))
                batch = list(dataloader)[0]
                _, partners_trajectories, _ = batch

                with torch.no_grad():
                    chooser_history, _, _ = env.play_full_episode(partners_trajectories)
                    probs, actions, rewards = chooser_history.get_full_trajectory()
                    targets = env.get_target_choices()

                accuracies.loc[len(accuracies)] = pd.Series({
                    'tau_fluc': np.round(tau_fluc, 2),
                    'tau_samp': np.round(tau_samp, 2),
                    'accuracy': accuracy_metric(actions, targets).item()
                })
                excess_rewards.loc[len(excess_rewards)] = pd.Series({
                    'tau_fluc': np.round(tau_fluc, 2),
                    'tau_samp': np.round(tau_samp, 2),
                    'excess_reward': excess_reward_metric(actions, partners_trajectories).item()
                })

            plot_heatmap(accuracies, axs[i][0])
            axs[i][0].set_title(f"Accuracies:\n{env.chooser.training_description}", fontsize=8)

            plot_heatmap(excess_rewards, axs[i][1])
            axs[i][1].set_title(f"Excess Reward:\n{env.chooser.training_description}", fontsize=8)

        plt.tight_layout()

        if self.save_dir is not None:
            plt.savefig(self.save_dir / 'uncertainty_generalization.png')

        if show:
            plt.show()
        else:
            plt.close()

    def trajectory_visualization(self, test_idx: int = 0, show: bool = True):
        test_set = BanditDataset.load(name='test', directory=DATA_DIR)
        test_dataloader = DataLoader(test_set, batch_size=len(test_set))
        batch = list(test_dataloader)[0]
        _, partners_trajectories, _ = batch

        test_set.plot(test_idx, show=False, simplified=True)
        for i, env in enumerate(self.envs):
            env.chooser.policy.eval()
            with torch.no_grad():
                chooser_history, _, _ = env.play_full_episode(partners_trajectories)
                probs, actions, rewards = chooser_history.get_full_trajectory()

            probs_to_plot = probs[test_idx].detach().numpy()
            plt.plot(list(range(len(probs_to_plot))), probs_to_plot, label=str(env.chooser.training_description))

        plt.ylabel('Bandit reward / Model Output Probability')
        plt.legend(bbox_to_anchor=(1, 0.5), loc="upper left")

        if self.save_dir is not None:
            plt.savefig(self.save_dir / 'trajectory_visualization.png')

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def _load_test_data():
        # Load test dataset
        test_set = BanditDataset.load(name='test', directory=DATA_DIR)
        test_dataloader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
        full_batch = list(test_dataloader)[0]
        _, partners_trajectories, _ = full_batch
        return partners_trajectories

    @staticmethod
    def _generate_test_trajectories(envs: List[Env], partners_trajectories: Tensor) -> List[dict]:
        trajectories: List[dict] = []

        for env in envs:
            if isinstance(env.chooser.policy, nn.Module):
                env.chooser.policy.eval()
            with torch.no_grad():
                chooser_history, _, _ = env.play_full_episode(partners_trajectories)
            probs, actions, rewards = chooser_history.get_full_trajectory()
            trajectory = {
                'description': env.chooser.training_description,
                'actions': actions,
                'probs': probs,
                'rewards': rewards,
                'targets': env.get_target_choices(),
                'partner_trajectories': partners_trajectories
            }
            trajectories.append(trajectory)
        return trajectories

