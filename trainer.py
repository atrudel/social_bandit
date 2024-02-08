import argparse
import subprocess
import sys
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader

from RNN import RNN
from config import DEVICE, DATA_DIR
from dataset import BanditDataset

parser = argparse.ArgumentParser(description="Training of RNN model.")

parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--hidden_size', type=int, default=128, help='Number of RNN hidden units')
parser.add_argument('--n_layers', type=int, default=2, help='Number of RNN layers')
parser.add_argument('--reward_loss', type=float, default=1, help='Coefficient of the reward maximizing objective in the overall loss function.')
parser.add_argument('--equity_loss', type=float, default=0, help='Coefficient of the equity maximization objective in the overall loss function.')
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Directory that contains the bandit trajectory data files.')
parser.add_argument('--debug', action='store_true', help='debug mode')
parser.add_argument('--seed', type=int, default=42, help='Random seed')


def launch_training(args: argparse.Namespace):
    commit = check_git_status() if not args.debug else None
    print(commit)

    # Set random seeds
    seed_everything(args.seed, workers=True)

    # Load datasets
    train_data = BanditDataset.load(name='train', directory=args.data_dir)
    val_data = BanditDataset.load(name='val', directory=args.data_dir)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    # Instantiate model
    model = RNN(learning_rate=args.lr,
                hidden_size=args.hidden_size,
                num_layers=args.n_layers,
                reward_loss_coef=args.reward_loss,
                equity_loss_coef=args.equity_loss,
                commit=commit,
                seed=args.seed)

    # Train model
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=DEVICE,
        fast_dev_run=True if args.debug else False,
        logger=True,
        deterministic=True
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    # Create a git tag on the current commit with the version name
    tag_version_in_git(trainer)
    return model


def check_git_status():
    # Check that the git working tree is clean otherwise exit the program
    git_status_output = subprocess.check_output(
        "git status --porcelain --untracked-files=no",
        shell=True, text=True
    )
    if git_status_output.strip():
        print("\033[31mCOMMIT all your changes before you run the training script.\033[0m")
        sys.exit(1)

    # Get the short commit hash
    commit_hash = subprocess.check_output(
        "git rev-parse --short=7 HEAD",
        shell=True,
        text=True
    ).strip()
    return commit_hash

def tag_version_in_git(trainer: Trainer):
    logs_folder = None
    for callback in trainer.callbacks:
        from lightning.pytorch.callbacks import ModelCheckpoint
        if isinstance(callback, ModelCheckpoint):
            logs_folder = callback.dirpath
            break
    if logs_folder is not None:
        version_name = Path(logs_folder).parts[-2]
        try:
            subprocess.run(["git", "tag", version_name], check=True)
            print(version_name)
        except subprocess.CalledProcessError as e:
            print(f"Unable to create git tag with version name: {e}")
    else:
        print("Unable to retrieve version name")


if __name__ == '__main__':
    args = parser.parse_args()
    launch_training(args)
