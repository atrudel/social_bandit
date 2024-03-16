from __future__ import annotations

import argparse
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta
from tqdm import tqdm

from social_bandit.config import N_TRIALS, TAU_FLUC, TAU_SAMP, EPIMIN, EPIMAX, NEPI, DATA_DIR, GENERALIZATION_TAU_FLUCS, \
    GENERALIZATION_TAU_SAMPS, GENERALIZATION_SET_SIZE
from social_bandit.data_generation.dataset import BanditDataset


parser = argparse.ArgumentParser(description="Generation of Bandit trajectories.")

parser.add_argument('--n_train', type=int, default=100000, help='number of training examples')
parser.add_argument('--n_val', type=int, default=10000, help='number of validation examples')
parser.add_argument('--n_test', type=int, default=1000, help='number of test examples')
parser.add_argument('--length', type=int, default=N_TRIALS, help='length of the trial sequences')
parser.add_argument('--tau_fluc', type=float, default=TAU_FLUC, help='temperature of the fluctuation of the mean')
parser.add_argument('--tau_samp', type=float, default=TAU_SAMP, help='temperature of the sampling')
parser.add_argument('--epimin', type=int, default=EPIMIN, help='mininum length of an episode')
parser.add_argument('--epimax', type=int, default=EPIMAX, help='maximum length of an episode')
parser.add_argument('--nepi', type=int, default=NEPI, help='number of episodes')
parser.add_argument('--seed', type=int, default=1, help='initial random state of the generator')
parser.add_argument('--debug', action='store_true', help='debug mode, does not store data')


def beta_sample(p, t):
    a = 1 + p * math.exp(t)
    b = 1 + (1 - p) * math.exp(t)
    return beta.rvs(a, b)


class EpisodeGenerator:
    def __init__(self, min_length, max_length, tau_fluc):
        self.min_length = min_length
        self.max_length = max_length
        self.tau_fluc = tau_fluc
        self.episodes = None

    def _generate_episodes(self, number):
        episodes = [self._generate_episode(self.min_length, self.max_length) for _ in range(number)]
        return episodes

    def _generate_episode(self, min_length, max_length):
        # Loop until a valid episode is generated
        while True:
            # Generate first mean of the episode such that > 0.5
            episode = []
            first_mean = beta_sample(0.5, self.tau_fluc)
            first_mean = 1 - first_mean if first_mean < 0.5 else first_mean
            episode.append(first_mean)

            for _ in range(max_length):
                new_mean = beta_sample(episode[-1], self.tau_fluc)
                # Stop generation if the mean fell below 0.5
                if new_mean < 0.5:
                    break
                else:
                    episode.append(new_mean)

            # Return episode if length is appropriate
            if min_length <= len(episode) <= max_length:
                return np.array(episode)


class BanditGenerator:
    def __init__(self,
                 tau_fluc=TAU_FLUC,
                 tau_samp=TAU_SAMP,
                 epimin=EPIMIN,
                 epimax=EPIMAX,
                 nepi=NEPI,
                 verbose=True):
        self.episode_generator = EpisodeGenerator(min_length=epimin, max_length=epimax, tau_fluc=tau_fluc)
        self.tau_samp = tau_samp
        self.nepi = nepi
        self.verbose = verbose

    def generate_dataset(self, size, name=None, length=N_TRIALS) -> BanditDataset:
        # Generate 'batch_size' pairs of bandit trajectories
        means = self._generate_latent_means(length, size)
        values = self._sample_values(means)
        dataset = BanditDataset(means=means, values=values, name=name)
        return dataset

    def _generate_latent_means(self, length: int, batch_size: int):
        trajectories = np.array([self._generate_trajectory(length, self.nepi)
                                 for _ in tqdm(range(batch_size * 2),
                                               desc="Generating trajectories",
                                               disable=not self.verbose)
                                 ])
        # Arrange bandits by pairs
        paired_trajectories = trajectories.reshape(batch_size, 2, length)
        # Flip Bandit no 2 for every sequence
        paired_trajectories[:, 1, :] = 1 - paired_trajectories[:, 1, :]
        return paired_trajectories

    def _generate_trajectory(self, length, num_episodes):
        trajectory = np.array([])
        while len(trajectory) < length:
            episodes = self.episode_generator._generate_episodes(num_episodes)
            # Flip every odd-indexed episode so that it goes below 0.5
            for i in range(num_episodes):
                if i % 2 == 1:
                    episodes[i] = 1 - episodes[i]
            trajectory = np.concatenate(episodes, axis=0)
        return trajectory[:length]

    def _sample_values(self, means):
        values = beta_sample(means, self.tau_samp)
        return values


class GeneralizationDatasetBundle:
    dump_filename = 'uncertainty_generalization.pickle'
    def __init__(self, tau_flucs: np.ndarray, tau_samps: np.ndarray):
        self.tau_flucs = tau_flucs
        self.tau_samps = tau_samps
        self.datasets = defaultdict(dict)

    def generate_datasets(self, size: int) -> GeneralizationDatasetBundle:
        progress_bar = tqdm(total=len(self.tau_flucs) * len(self.tau_samps),
                            desc=f'Generating trajectories for evaluation sets')
        for tau_fluc in self.tau_flucs:
            for tau_samp in self.tau_samps:
                data_generator = BanditGenerator(tau_fluc=tau_fluc, tau_samp=tau_samp, verbose=False)
                dataset: BanditDataset = data_generator.generate_dataset(size=size,
                                                                         name=self._format_filename(tau_fluc, tau_samp),
                                                                         length=N_TRIALS)
                self.datasets[tau_fluc][tau_samp] = dataset
                progress_bar.update(1)
        return self

    def get(self, tau_fluc: float, tau_samp: float) -> BanditDataset:
        return self.datasets[tau_fluc][tau_samp]

    def save(self, directory: str = DATA_DIR) -> None:
        save_dir = Path(directory) / self.dump_filename
        if not self.datasets:
            raise Exception("GeneralizationDatasetBundle error: you must generate datasets before saving them.")

        with open(save_dir, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, directory: str = DATA_DIR) -> GeneralizationDatasetBundle:
        with open(Path(directory) / cls.dump_filename, 'rb') as f:
            return pickle.load(f)

    def __iter__(self):
        for tau_fluc, samps in self.datasets.items():
            for tau_samp, dataset in samps.items():
                yield tau_fluc, tau_samp, dataset

    def _format_filename(self, tau_fluc, tau_samp) -> str:
        return f"tau_fluc={tau_fluc:.2g}__tau_samp={tau_samp:.2g}"


if __name__ == '__main__':
    args = parser.parse_args()

    np.random.seed(args.seed)
    generator = BanditGenerator(
        args.tau_fluc,
        args.tau_samp,
        args.epimin,
        args.epimax,
        args.nepi,
    )
    train_dataset = generator.generate_dataset(args.n_train, name='train', length=args.length)

    if not args.debug:
        os.makedirs(DATA_DIR, exist_ok=True)

        val_dataset: BanditDataset = generator.generate_dataset(args.n_val, name='val', length=args.length)
        test_dataset: BanditDataset = generator.generate_dataset(args.n_test, name='test', length=args.length)
        uncertainty_generation_datasets = GeneralizationDatasetBundle(GENERALIZATION_TAU_FLUCS,
                                                                      GENERALIZATION_TAU_SAMPS).generate_datasets(GENERALIZATION_SET_SIZE)

        train_dataset.save(DATA_DIR)
        val_dataset.save(DATA_DIR)
        test_dataset.save(DATA_DIR)
        uncertainty_generation_datasets.save(DATA_DIR)

        print(f"Bandit trajectories generated. ({args.n_train} train, {args.n_val} val, {args.n_test} test, length={args.length})")
