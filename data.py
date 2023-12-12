import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
from torch.utils.data import Dataset


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
        self.values = np.load(filename)

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
    tau_fluc = 3
    tau_samp = 2
    length = 80

    generator = UnrestrictedTrajectoryGenerator(tau_fluc, tau_samp, random_state=42)
    train_set = generator.generate_batch(length, 10000)
    test_set = generator.generate_batch(length, 100)

    np.save('train', train_set)
    np.save('test', test_set)
    print("Data generated.")
