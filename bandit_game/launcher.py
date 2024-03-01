import argparse
import glob

from data_generation.dataset import BanditDataset
from bandit_game.game import Game
from bandit_game.partner import Partner
from bandit_game.player import Player
from bandit_game.strategies import RNNStrategy

parser = argparse.ArgumentParser(description="Launch a social bandit bandit_game.")

parser.add_argument('--data_dir', type=str, default='.',
                    help='Directory containing bandit trajectory data files')
parser.add_argument('--trajectory_no', type=int, required=False, default=None,
                    help='Index of the trajectory to play. If specified, only one trajectory will be executed and visualized.')
parser.add_argument('--experiment_name', type=str,
                    help='Name of the directory within lightning_logs that contains the trained model')


if __name__ == '__main__':
    args = parser.parse_args()

    test_dataset = BanditDataset.load(name='test', directory=args.data_dir)

    if args.trajectory_no is not None:
        _, bandit_trajectories, _ = test_dataset[args.trajectory_no]  # dims: batch, 2, length

        checkpoint_path: str = glob.glob(f"lightning_logs/{args.experiment_name}/checkpoints/*.ckpt")[-1]
        player = Player(RNNStrategy(checkpoint_path))
        partner1 = Partner(bandit_trajectories[0].reshape(1, -1))
        partner2 = Partner(bandit_trajectories[1].reshape(1, -1))
        game = Game(player, partner1, partner2)

        game.visualize_one_round(args.trajectory_no)