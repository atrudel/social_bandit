import argparse
import concurrent
import math
from itertools import repeat

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
from torch.utils.data import Dataset
from tqdm import tqdm

from config import SEQUENCE_LENGTH, TAU_FLUC, TAU_SAMP, EPIMIN, EPIMAX, NEPI

parser = argparse.ArgumentParser(description="Generation of Bandit trajectories.")

parser.add_argument('--n_train', type=int, default=100000, help='number of training examples')
parser.add_argument('--n_val', type=int, default=10000, help='number of validation examples')
parser.add_argument('--n_test', type=int, default=1000, help='number of test examples')
parser.add_argument('--length', type=int, default=SEQUENCE_LENGTH, help='length of the trial sequences')
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
    def __init__(self, min_length, max_length, tau_fluc, seed=None):
        self.min_length = min_length
        self.max_length = max_length
        self.tau_fluc = tau_fluc
        self.seed = seed
        self.episodes = None

    def reset(self, episode_pool_size):
        # self.episodes = self._generate_episodes(episode_pool_size)
        pass

    def sample_episodes(self, n):
        # if self.episodes is None:
        #     raise RuntimeError("Call the reset() method on EpisodeGenerator before you call sample_episodes().")
        # idxs = np.random.choice(len(self.episodes), size=n)
        # return [self.episodes[i] for i in idxs]
        return self._generate_episodes(n)

    def _generate_episodes(self, number):
        np.random.seed(self.seed)
        # episodes = []
        # for _ in tqdm(range(number), desc='Generating episodes'):
        #     episodes.append(self._generate_episode(self.min_length, self.max_length))
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
                 seed=None):
        self.episode_generator = EpisodeGenerator(min_length=epimin, max_length=epimax, tau_fluc=tau_fluc, seed=seed)
        self.tau_samp = tau_samp
        self.nepi = nepi
        self.seed = seed

    def generate_batch(self, batch_size, length=SEQUENCE_LENGTH):
        # Generate 'batch_size' pairs of bandit trajectories
        self.episode_generator.reset(episode_pool_size=batch_size * 10)
        self.means = self._generate_latent_means(length, batch_size)
        self.values = self._sample_values(self.means)
        return self.values

    def _generate_latent_means(self, length: int, batch_size: int):
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     trajectories = list(tqdm(executor.map(
        #         self._generate_trajectory, repeat(length, batch_size * 2), repeat(self.nepi, batch_size * 2)),
        #         total=batch_size * 2, desc="Generating trajectories"
        #     ))
        # trajectories = np.array(trajectories)
        trajectories = np.array([self._generate_trajectory(length, self.nepi)
                                 for _ in tqdm(range(batch_size * 2), desc="Generating trajectories")])
        # Arrange bandits by pairs
        paired_trajectories = trajectories.reshape(batch_size, 2, length)
        # Flip Bandit no 2 for every sequence  # Todo check this is necessary
        paired_trajectories[:, 1, :] = 1 - paired_trajectories[:, 1, :]
        return paired_trajectories

    def _generate_trajectory(self, length, num_episodes):
        episodes = self.episode_generator.sample_episodes(num_episodes)
        # Flip every odd-indexed episode so that it goes below 0.5
        for i in range(num_episodes):
            if i % 2 == 1:
                episodes[i] = 1 - episodes[i]
        trajectory = np.concatenate(episodes, axis=0)

        if len(trajectory) < length:
            return self._generate_trajectory(length, num_episodes)
        else:
            return trajectory[:length]

    def _sample_values(self, means):
        values = beta_sample(means, self.tau_samp)
        return values


class BanditDataset(Dataset):
    def __init__(self, filename=None, values=None):
        if values is None:
            self.values = np.load(filename).astype('float32')
        else:
            self.values = values.astype('float32')

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, item):
        bandits = self.values[item]
        target = np.argmax(bandits, axis=0).astype('float32')
        return bandits, target

    def plot(self, item, comment=None):
        bandits, target = self[item]
        traj1 = bandits[0]
        traj2 = bandits[1]
        plt.plot(traj1, label="Bandit 0")
        plt.plot(traj2, label="Bandit 1")
        plt.scatter(list(range(len(traj1))), target, label="target")
        plt.title(f"Bandit trajectories (no {item})" + (f" - {comment}" if comment is not None else ""))
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Reward")
        plt.show()


if __name__ == '__main__':
    args = parser.parse_args()

    generator = BanditGenerator(
        args.tau_fluc,
        args.tau_samp,
        args.epimin,
        args.epimax,
        args.nepi,
        seed=args.seed
    )
    train_set = generator.generate_batch(args.n_train, args.length)

    if not args.debug:
        val_set = generator.generate_batch(args.n_val, args.length)
        test_set = generator.generate_batch(args.n_test, args.length)

        np.save('train', train_set)
        np.save('val', val_set)
        np.save('test', test_set)
        print(f"Bandit trajectories generated. ({args.n_train} train, {args.n_val} val, {args.n_test} test, length={args.length})")
