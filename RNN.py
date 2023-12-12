import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy


class RNN(L.LightningModule):
    def __init__(self,
                 learning_rate,
                 hidden_size,
                 num_layers = 1,
                 first_choice = 0
                 ):
        super(RNN, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.first_choice = first_choice

        self.rnn = nn.RNN(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        ).double()
        self.fc = nn.Linear(hidden_size, 1).double()
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.accuracy = BinaryAccuracy()

    def forward(self, input):
        out, _ = self.rnn(input)
        out = out[:, -1, :]  # batch, seq_len, hidden_size
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

    def training_step(self, batch, batch_idx):
        self.train()
        choices, seq_len, targets, trajectories = self._prepare_prediction_variables(batch)
        loss = 0

        for i in range(1, seq_len):
            out_trial = self._predict_one_trial(choices, i, trajectories)
            target_trial = targets[:, i].reshape(-1, 1)

            opt = self.optimizers()
            opt.zero_grad()
            loss_trial = self.criterion(out_trial, target_trial)
            self.manual_backward(loss_trial)
            opt.step()

            loss += loss_trial
            choices[:,i] = out_trial.round().squeeze()

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def _prepare_prediction_variables(self, batch):
        trajectories, targets = batch
        seq_len = trajectories.shape[2]
        batch_size = trajectories.shape[0]
        choices = torch.zeros((batch_size, seq_len), dtype=torch.int64)
        choices[:, 0] = self.first_choice  # First choice is pre-decided
        targets = targets.double()
        return choices, seq_len, targets, trajectories

    def _predict_one_trial(self, choices, i, trajectories):
        rewards = torch.gather(trajectories, 1, choices.unsqueeze(1)).squeeze()  # batch_size, seq_len
        input = torch.stack([choices, rewards], dim=2)[:, :i, :]  # batch_size, seq_len, n_features
        out_trial = self(input)
        return out_trial

    def validation_step(self, batch, batch_idx):
        self.eval()
        choices, seq_len, targets, trajectories = self._prepare_prediction_variables(batch)

        for i in range(1, seq_len):
            out_trial = self._predict_one_trial(choices, i, trajectories)
            choices[:,i] = out_trial.round().squeeze()

        loss = self.criterion(choices.double(), targets)
        accuracy = self.accuracy(choices, targets)
        self.log_dict({'val_loss': loss, 'val_acc': accuracy}, prog_bar=True)
        return accuracy


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )
        return optimizer