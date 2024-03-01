from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from config import DATA_DIR


FILE_SUFFIX = '_dataset.npy'

class BanditDataset(Dataset):
    def __init__(self, means: np.ndarray, values: np.ndarray, name: str = None):
        self.means: np.ndarray = means.astype('float32')
        self.values: np.ndarray = values.astype('float32')
        self.name: Optional[str] = name

    def save(self, directory: str) -> None:
        data = np.concatenate([
            np.expand_dims(self.means, 3),
            np.expand_dims(self.values, 3),
        ], axis=3)
        if self.name is None:
            raise ValueError("An unnamed dataset cannot be saved.")
        np.save(Path(directory) / f"{self.name}{FILE_SUFFIX}", data)

    @classmethod
    def load(cls, name: str, directory: str = DATA_DIR) -> BanditDataset:
        data = np.load(Path(directory) / f"{name}{FILE_SUFFIX}")
        means = data[:, :, :, 0].astype('float32')
        values = data[:, :, :, 1].astype('float32')
        return cls(means, values, name=name)

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, item):
        means = self.means[item]
        values = self.values[item]
        target = np.argmax(values, axis=0).astype('float32')
        return means, values, target

    def plot(self, item, comment=None, show=True, simplified=False):
        means, values, target = self[item]
        if simplified:
            plt.plot(values[0], label="Bandit 0: values", color="tab:blue", linestyle='dashed', linewidth=2, alpha=0.7)
            plt.plot(values[1], label="Bandit 1: values", color="orange", linestyle='dashed', linewidth=2, alpha=0.7)
        else:
            plt.plot(values[0], label="Bandit 0: reward values", color="tab:blue")
            plt.plot(values[1], label="Bandit 1: reward values", color="orange")
            plt.plot(means[0], label="Bandit 0: latent mean", color="tab:cyan", linestyle="dotted")
            plt.plot(means[1], label="Bandit 1: latent mean", color="tan", linestyle="dotted")
            plt.scatter(list(range(len(values[0]))), target, label="target")
        plt.title(f"Bandit trajectories (no {item})" + (f" - {comment}" if comment is not None else ""))
        plt.legend(bbox_to_anchor=(1, 0.5), loc="upper left")
        plt.xlabel("Time step")
        plt.ylabel("Reward")
        if show:
            plt.show()


if __name__ == '__main__':
    dataset = BanditDataset.load('train')
    dataset.plot(0)