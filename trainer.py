import argparse

from torch.utils.data import DataLoader
from lightning import Trainer

from RNN import RNN
from config import DEVICE
from data import BanditDataset

parser = argparse.ArgumentParser(description="Training of RNN model.")

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--hidden_size', type=int, default=128, help='number of RNN hidden units')
parser.add_argument('--n_layers', type=int, default=2, help='number of RNN layers')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log_dir', type=str, default=None)


def launch_training(args: argparse.Namespace):
    model = RNN(args.lr, args.hidden_size, args.n_layers)

    train_data = BanditDataset('train.npy')
    val_data = BanditDataset('test.npy')

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    trainer = Trainer(default_root_dir=args.log_dir, accelerator=DEVICE)
    trainer.fit(model, train_dataloader, val_dataloader)



if __name__ == '__main__':
    args = parser.parse_args()
    launch_training(args)
