import os
import pickle
import random
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from social_bandit import config
from social_bandit.game.chooser import Chooser
from social_bandit.game.environment import Env
from social_bandit.game.partner import DataPartner
from social_bandit.data_generation.dataset import BanditDataset
from social_bandit.evaluation.metrics import accuracy_metric, excess_reward_metric
from social_bandit.models.rnn_chooser import RNNChooserPolicy
from social_bandit.training.objective_functions import ObjectiveFunction, MeanRewardObjectiveFunction


class Trainer:
    def __init__(self,
                 env: Env,
                 trained_model: nn.Module,
                 objective_function: ObjectiveFunction,
                 training_name: str,
                 experiment_dir: Path = config.EXPERIMENT_DIR / 'default_experiment',
                 data_dir: Path = config.DATA_DIR,
                 validate_every_n_steps: int = config.VALIDATE_EVERY_NSTEPS
                 ):
        self.env: Env = env
        self.trained_model: nn.Module = trained_model
        self.optimizer: Optional[Optimizer] = None
        self.objective_function: ObjectiveFunction = objective_function
        self.validate_every_n_steps: int = validate_every_n_steps
        self.training_name: str = training_name
        self.save_dir: Path = experiment_dir / training_name
        self.data_dir: Path = data_dir
        self.writer: Optional[SummaryWriter] = None

    def launch_training(self, n_epochs: int,
                        seed: int,
                        learning_rate: float,
                        batch_size: int = config.BATCH_SIZE,
                        debug: bool = False):
        if not debug:
            os.makedirs(self.save_dir)
            self.writer = SummaryWriter(self.save_dir / 'tensorboard_logs')
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_data = BanditDataset.load(name='train', directory=self.data_dir)
        val_data = BanditDataset.load(name='val', directory=self.data_dir)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)

        self.optimizer = torch.optim.Adam(self.trained_model.parameters(), lr=learning_rate)
        self._training_loop(n_epochs, train_dataloader, val_dataloader, debug)

    def _training_loop(self, n_epochs: int, train_dataloader: DataLoader, val_dataloader: DataLoader, debug: bool = False):
        best_val_loss = float('inf')

        for i_epoch in range(n_epochs):
            with tqdm(train_dataloader, unit='batch') as progress_bar:
                progress_bar.set_description(f"Epoch {i_epoch}")

                for j_batch, data_batch in enumerate(progress_bar):
                    global_step = i_epoch * len(train_dataloader) + j_batch
                    _, partner_trajectories, _ = data_batch

                    chooser_trajectory, _, _ = self.env.play_full_episode(partner_trajectories)
                    probs, actions, rewards = chooser_trajectory.get_full_trajectory()
                    train_loss = self._training_step(probs, actions, rewards)

                    if not debug:
                        self.writer.add_scalar('Loss/train', train_loss, global_step)

                    if global_step % self.validate_every_n_steps == 0:
                        val_metrics = self._validation_step(val_dataloader)
                        val_loss, val_accuracy, val_excess_reward = val_metrics

                        # Log to progress bar
                        progress_bar.set_postfix({
                            'Loss/train': train_loss,
                            'Loss/val': val_loss,
                            'Accuracy/val': val_accuracy,
                            'Excess reward/val': val_excess_reward
                        })
                        # Log to tensorboard
                        if not debug:
                            self.writer.add_scalar('Loss/val', val_loss, global_step)
                            self.writer.add_scalar('Accuracy/val', val_accuracy, global_step)
                            self.writer.add_scalar('Excess_rwd/val', val_excess_reward, global_step)

                        # Save best model
                        if val_loss < best_val_loss:
                            self.trained_model.save(self.save_dir / 'best_model.pth')
                            best_val_loss = val_loss

        self.trained_model.save(self.save_dir / 'final_model.pth')

    def _training_step(self, probs: Tensor, actions: Tensor, rewards: Tensor) -> float:
        """Performs one optimization step based on the actions and rewards of an episode"""
        self.trained_model.train()
        self.optimizer.zero_grad()
        loss = self.objective_function.compute_loss(probs, actions, rewards)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _validation_step(self, val_dataloader: DataLoader) -> Tuple[float, float, float]:
        self.trained_model.eval()
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

        return avg_loss.item(), avg_accuracy.item(), avg_excess_reward.item()

    def save_models(self):
        with open(self.save_dir / 'description.txt', 'w') as file:
            file.write(self)
        with open(self.save_dir / 'chooser', 'rb') as file:
            pickle.dump(self.env.chooser, file)
        with open(self.save_dir / 'partner_0', 'wb') as file:
            pickle.dump(self.env.partner_0, file)
        with open(self.save_dir / 'partner_1', 'wb') as file:
            pickle.dump(self.env.partner_1, file)

    def __repr__(self) -> str:
        return f"""Training: {self.training_name}
        
        Model:
        {self.trained_model}
        
        Objective function:
        {self.objective_function}
        
        Training parameters:
        - num_epochs: {self}
        
        =ENV====================
        Chooser:
        {self.env.chooser}
        
        Partner 0:
        {self.env.partner_0}
        
        Partner 1:
        {self.env.partner_1}
        """


if __name__ == '__main__':
    policy_model = RNNChooserPolicy(hidden_size=128, n_layers=2)
    env = Env(
        chooser=Chooser(policy_model),
        partner_0=DataPartner(),
        partner_1=DataPartner()
    )
    trainer = Trainer(env=env,
                      trained_model=policy_model,
                      objective_function=MeanRewardObjectiveFunction(),
                      training_name='test_training_save',
                      experiment_dir=Path(config.EXPERIMENT_DIR) / 'test_exp',
                      validate_every_n_steps=1,
                      )
    trainer.launch_training(
        n_epochs=1,
        seed=10,
        learning_rate=1e-3,
        debug=True
    )