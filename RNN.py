import lightning as L
import torch
import torch.nn as nn
from torchtyping import TensorType

from metrics import accuracy, excess_reward


class RNN(L.LightningModule):
    def __init__(self,
             learning_rate: float,
             hidden_size: int,
             num_layers: int,
             commit: str = None,
             first_choice: int = 0  # Action to always choose on first trial (0 or 1)
             ):
        super(RNN, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.first_choice = first_choice
        self.first_prob = 0.5

        self.rnn = nn.RNN(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, h_0):
        out, h_n = self.rnn(input, h_0)
        out = out[:, -1, :]  # [batch, seq_len, hidden_size], taking the last hidden state of the sequence
        out = self.fc(out)
        out = self.sigmoid(out)
        return out, h_n

    def training_step(self, batch, batch_idx):
        self.train()
        actions, probs, rewards, _, _ = self.process_trajectory(batch)

        # Compute loss on sequence and optimize
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.criterion(rewards, probs, actions)
        self.manual_backward(loss)
        opt.step()

        self.log('train_loss', loss.item(), prog_bar=True)
        return loss

    def process_trajectory(self, batch):
        # Initialize sequential variables from batch and pre-fill with the default first action
        probs, actions, rewards, seq_len, targets, trajectories = self._prepare_prediction_variables(batch)
        action = actions[:, 0].reshape(-1, 1)
        reward = rewards[:, 0].reshape(-1, 1)
        hidden_state = None
        # Iterate over sequence length, skipping first action
        for i_trial in range(1, seq_len):
            action, out_prob, hidden_state = self._choose_next_action(action, reward, i_trial, hidden_state)
            reward = self._play_action(action, i_trial, trajectories)

            # Store data in sequence history
            probs[:, i_trial] = out_prob[:, 0]
            actions[:, i_trial] = action[:, 0]
            rewards[:, i_trial] = reward[:, 0]
        return actions, probs, rewards, targets, trajectories

    def _prepare_prediction_variables(self, batch):
        _, trajectories, targets = batch  # Ignoring latent means for now
        seq_len = trajectories.shape[2]
        batch_size = trajectories.shape[0]

        actions = torch.zeros((batch_size, seq_len), dtype=torch.int64)
        actions[:, 0] = self.first_choice  # First choice is pre-decided

        probs = torch.zeros_like(actions, dtype=torch.float)
        probs[:, 0] = self.first_prob

        rewards = torch.zeros_like(actions, dtype=torch.float)
        rewards[:, 0] = trajectories[:, self.first_choice, 0]


        return probs, actions, rewards, seq_len, targets, trajectories

    def _choose_next_action(self, action, reward, i_trial, hidden_state):
        # Prepare RNN input
        input = torch.stack([action, reward], dim=2)[:, :i_trial, :]  # batch_size, seq_len, n_features

        # Apply RNN on sequence of choices and rewards
        out_prob, hidden_state = self(input, hidden_state)

        # Sample action from output probability
        action = torch.bernoulli(out_prob).long()
        return action, out_prob, hidden_state

    def _play_action(self, action, i_trial, trajectories) -> TensorType["batch", 1, float]:
        reward = torch.gather(trajectories[:, :, i_trial], 1, action)
        return reward

    def criterion(self, rewards, probs, actions):
        mean_rewards = rewards.mean(dim=1).unsqueeze(1)
        deltas = rewards - mean_rewards
        corrected_probs = probs * actions + (1 - probs) * (1 - actions)
        losses = -deltas * corrected_probs
        return losses.sum()

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            actions, probs, rewards, targets, trajectories = self.process_trajectory(batch)

        loss = self.criterion(rewards, probs, actions)
        acc = accuracy(actions, targets)
        excess_rwd = excess_reward(actions, trajectories)

        self.log_dict({
            'val_loss': loss.item(),
            'val_acc': acc.item(),
            'val_excess_rwd': excess_rwd.item()
        }, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )
        return optimizer