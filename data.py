import argparse
import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
from torch.utils.data import Dataset

parser = argparse.ArgumentParser(description="Generation of Bandit trajectories.")

parser.add_argument('--n_train', type=int, default=100000, help='number of training examples')
parser.add_argument('--n_test', type=int, default=10000, help='number of test examples')
parser.add_argument('--length', type=int, default=80, help='length of the trial sequences')
parser.add_argument('--tau_fluc', type=float, default=3, help='temperature of the fluctuation of the mean')
parser.add_argument('--tau_samp', type=float, default=2, help='temperature of the sampling')
parser.add_argument('--seed', type=int, default=42, help='initial random state of the generator')


class UnrestrictedTrajectoryGenerator:
    def __init__(self, tau_fluc, tau_samp, random_state=None):
        self.tau_fluc = tau_fluc
        self.tau_samp = tau_samp
        np.random.seed(random_state)

    def generate_batch(self, length, batch_size):
        # Generate 'batch_size' pairs of bandit trajectories
        self.means = self._generate_latent_means(length, batch_size * 2)
        self.values = self._sample_values(self.means)
        return self.values.reshape(batch_size, 2, length)

    def _beta_sample(self, p, t):
        a = 1 + p * math.exp(t)
        b = 1 + (1 - p) * math.exp(t)
        return beta.rvs(a, b)

    def _generate_latent_means(self, length: int, quantity: int):
        means = np.zeros((quantity, length))
        means[:, 0] = self._beta_sample(
            np.full(shape=quantity, fill_value=0.5),
            self.tau_fluc
        )
        for i in range(1, length):
            means[:,i] = self._beta_sample(means[:, i - 1], self.tau_fluc)
        return means

    def _sample_values(self, means):
        values = self._beta_sample(means, self.tau_samp)
        return values


class BanditDataset(Dataset):
    def __init__(self, filename):
        self.values = np.load(filename).astype('float32')

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, item):
        bandits = self.values[item]
        target = np.argmax(bandits, axis=0).astype('float32')
        return bandits, target

    def plot(self, item):
        bandits, target = self[item]
        traj1 = bandits[0]
        traj2 = bandits[1]
        plt.plot(traj1, label="Bandit 0")
        plt.plot(traj2, label="Bandit 1")
        plt.scatter(list(range(len(traj1))), target, label="target")
        plt.title(f"Bandit trajectories")
        plt.legend()
        plt.xlabel("Time step")
        plt.ylabel("Reward")
        plt.show()


if __name__ == '__main__':
    args = parser.parse_args()

    generator = UnrestrictedTrajectoryGenerator(args.tau_fluc, args.tau_samp, random_state=args.seed)
    train_set = generator.generate_batch(args.length, args.n_train)
    test_set = generator.generate_batch(args.length, args.n_test)

    np.save('train', train_set)
    np.save('test', test_set)
    print(f"Bandit trajectories generated. ({args.n_train} train, {args.n_test} test, length={args.length})")
