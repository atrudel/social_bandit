import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from bandit_game.chooser import Chooser, RNNChooserPolicy
from bandit_game.environment import Env
from bandit_game.partner import DataPartner
from data_generation.dataset import BanditDataset
from evaluation.metrics import accuracy_metric, excess_reward_metric
from models.rnn_chooser import RNNforBinaryAction
from training.objective_functions import ObjectiveFunction, RewardObjectiveFunction


class Trainer:
    def __init__(self,
                 env: Env,
                 model: nn.Module,
                 objective_function: ObjectiveFunction,
                 training_name: str,
                 experiment_dir: Path = config.EXPERIMENT_DIR / 'default_experiment',
                 data_dir: Path = config.DATA_DIR,
                 debug: bool = False):
        self.env: Env = env
        self.model: nn.Module = model
        self.optimizer: Optimizer = torch.optim.Adam(self.model.parameters(), )
        self.objective_function: ObjectiveFunction = objective_function
        self.training_dir: Path = experiment_dir / training_name
        log_dir = self.training_dir / 'tb_logs'
        self.writer: SummaryWriter = SummaryWriter(log_dir)
        self.data_dir: Path = data_dir
        self.debug: bool = debug

    def launch_training(self, n_epochs: int, seed: int, batch_size: int):
        os.makedirs(self.training_dir, exist_ok=self.debug)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_data = BanditDataset.load(name='train', directory=self.data_dir)
        val_data = BanditDataset.load(name='val', directory=self.data_dir)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)

        self._training_loop(n_epochs, train_dataloader, val_dataloader)

    def _training_loop(self, n_epochs: int, train_dataloader: DataLoader, val_dataloader: DataLoader):
        for i_epoch in range(n_epochs):
            with tqdm(train_dataloader, unit='batch') as progress_bar:
                progress_bar.set_description(f"Epoch {i_epoch}")
                for j_batch, data_batch in enumerate(train_dataloader):
                    global_step = i_epoch * len(train_dataloader) + j_batch
                    _, partner_trajectories, _ = data_batch
                    chooser_trajectory, _, _ = self.env.play_full_episode(partner_trajectories)
                    probs, actions, rewards = chooser_trajectory.get_full_trajectory()
                    train_loss = self._training_step(probs, actions, rewards, global_step)
                    val_metrics = self._validation_step(val_dataloader, global_step)

                    # Log to progress bar
                    val_loss, val_accuracy, val_excess_reward = val_metrics
                    progress_bar.set_postfix({
                        'Loss/train': train_loss,
                        'Loss/val': val_loss,
                        'Accuracy/val': val_accuracy,
                        'Excess reward/val': val_excess_reward
                    })

    def _training_step(self, probs: Tensor, actions: Tensor, rewards: Tensor, global_step: int) -> None:
        """Performs one optimization step based on the actions and rewards of an episode
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.objective_function.compute_loss(probs, actions, rewards)
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar('Loss/train', loss.item(), global_step)
        return loss.item()

    def _validation_step(self, val_dataloader: DataLoader, global_step: int) -> Tuple[float, float, float]:
        self.model.eval()
        losses = []
        accuracies = []
        excess_rewards = []
        with torch.no_grad():
            for data_batch in val_dataloader:
                _, partner_trajectories, _ = data_batch
                chooser_trajectory, _, _ = self.env.play_full_episode(partner_trajectories)
                probs, actions, rewards = chooser_trajectory.get_full_trajectory()
                targets = self.env.get_target_choices()
                # Compute loss
                loss = self.objective_function.compute_loss(probs, actions, rewards)
                losses.append(loss)
                # Compute metrics
                accuracy = accuracy_metric(actions, targets)
                accuracies.append(accuracy)
                excess_rwd = excess_reward_metric(actions, partner_trajectories)
                excess_rewards.append(excess_rwd)
        # Compute average metrics
        avg_loss = sum(losses) / len(losses)
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_excess_reward = sum(excess_rewards) / len(excess_rewards)

        # Log to tensorboard
        self.writer.add_scalar('Loss/val', avg_loss.item(), global_step)
        self.writer.add_scalar('Accuracy/val', avg_accuracy.item(), global_step)
        self.writer.add_scalar('Excess_rwd/val', avg_excess_reward.item(), global_step)

        return avg_loss.item(), avg_accuracy.item(), avg_excess_reward.item()



if __name__ == '__main__':
    model = RNNforBinaryAction(hidden_size=48, num_layers=1)
    policy_model = RNNChooserPolicy(model)
    env = Env(
        chooser=Chooser(policy_model),
        partner_0=DataPartner(),
        partner_1=DataPartner()
    )
    trainer = Trainer(env=env,
                      model=model,
                      objective_function=RewardObjectiveFunction(discount_factor=0.5),
                      training_name='test_training',
                      experiment_dir=Path(config.EXPERIMENT_DIR) / 'test_exp',
                      debug=True
                      )
    trainer.launch_training(
        n_epochs=3,
        seed=10,
        batch_size=10
    )