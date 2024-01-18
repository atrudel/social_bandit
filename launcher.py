import argparse
import glob

from data import BanditDataset
from game.game import Game
from game.partner import Partner
from game.player import Player
from game.strategies import RNNStrategy

parser = argparse.ArgumentParser(description="Launch a social bandit game.")

parser.add_argument('--file', type=str, default='test.npy',
                    help='File containing bandit trajectories')
parser.add_argument('--trajectory_no', type=int, required=False, default=None,
                    help='Index of the trajectory to play. If specified, only one trajectory will be executed and visualized.')
parser.add_argument('--experiment_name', type=str,
                    help='Name of the directory within lightning_logs that contains the trained model')


if __name__ == '__main__':
    args = parser.parse_args()

    test_dataset = BanditDataset(args.file)

    if args.trajectory_no is not None:
        bandit_trajectories, _ = test_dataset[args.trajectory_no]  # dims: batch, 2, length

        checkpoint_path: str = glob.glob(f"lightning_logs/{args.experiment_name}/checkpoints/*.ckpt")[-1]
        player = Player(RNNStrategy(checkpoint_path))
        partner1 = Partner(bandit_trajectories[0].reshape(1, -1))
        partner2 = Partner(bandit_trajectories[1].reshape(1, -1))
        game = Game(player, partner1, partner2)

        game.visualize_one_round(args.trajectory_no)