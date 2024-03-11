from argparse import Namespace

from social_bandit.config import DATA_DIR
from trainer import launch_training


parameters = {
    'lr': 1e-3,
    'hidden_size': 48,
    'n_layers': 2,
    'reward_loss': [0, 0.25, 0.5, 0.75, 1],
    'equity_loss': [1, 0.75, 0.5, 0.25, 0],
    'batch_size': 1000,
    'epochs': 200,
    'seed': 42,
    'data_dir': DATA_DIR,
    'debug': False
}


def launch_experiment(params):
    max_length = 1
    for param in params.values():
        if isinstance(param, list):
            if max_length != 1 and len(param) != max_length:
                raise ValueError('Parameter value lists must have the same length')
            max_length = len(param)

    for i in range(max_length):
        training_params = {
            key: value[i] if isinstance(value, list) else value
            for key, value in params.items()
        }
        launch_training(Namespace(**training_params))



if __name__ == '__main__':
    launch_experiment(parameters)