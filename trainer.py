import argparse

from torch.utils.data import DataLoader
from lightning import Trainer

from RNN import RNN
from config import DEVICE, DATA_DIR
from data_generator import BanditDataset

parser = argparse.ArgumentParser(description="Training of RNN model.")

parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--hidden_size', type=int, default=128, help='Number of RNN hidden units')
parser.add_argument('--n_layers', type=int, default=2, help='Number of RNN layers')
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Directory that contains the bandit trajectory data files.')
parser.add_argument('--commit', type=str, default=None, help='current commit hash of the code being run')
parser.add_argument('--debug', action='store_true', help='debug mode')


def launch_training(args: argparse.Namespace):
    model = RNN(args.lr, args.hidden_size, args.n_layers, commit=args.commit)

    train_data = BanditDataset('train', args.data_dir)
    val_data = BanditDataset('val', args.data_dir)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=DEVICE,
        fast_dev_run=True if args.debug else False,
        logger=False if args.debug else True
    )
    trainer.fit(model, train_dataloader, val_dataloader)



if __name__ == '__main__':
    args = parser.parse_args()
    launch_training(args)
