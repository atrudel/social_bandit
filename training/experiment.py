import os
from pathlib import Path

from .. import config
from bandit_game.chooser import Chooser, RNNChooserPolicy
from bandit_game.environment import Env
from bandit_game.partner import DataPartner
from models.rnn_chooser import RNNforBinaryAction
from training.objective_functions import RewardObjectiveFunction
from training.trainer import Trainer


class Experiment:
    def __init__(self, **params):
        self.training_params = self._pack_training_params(params)

    def launch_experiment(self, name: str):
        experiment_dir: Path = self._create_folder(name)

        for i, params in enumerate(self.training_params):
            model = RNNforBinaryAction(
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers']
            )
            objective_function = RewardObjectiveFunction(discount_factor=params['discount_factor'])
            env = Env(
                chooser=Chooser(RNNChooserPolicy(model)),
                partner_0=DataPartner(),
                partner_1=DataPartner()
            )
            trainer = Trainer(
                env=env,
                model=model,
                objective_function=objective_function,
                training_name=f"{i:02}_training",
                experiment_dir=experiment_dir,
                data_dir=config.DATA_DIR,
            )
            trainer.launch_training(
                n_epochs=params['n_epochs'],
                seed=params['seed'],
                batch_size=config.BATCH_SIZE
            )
            trainer.save()

    def _pack_training_params(self, params):
        num_trainings = self._count_number_of_trainings(params)
        training_configs = []
        for i in range(num_trainings):
            config = {}
            for key, value in params.items():
                if isinstance(value, list):
                    config[key] = value[i]
                else:
                    config[key] = value
            training_configs.append(config)
        return training_configs


    def _count_number_of_trainings(self, params):
        max_param_values = None
        for value in params.values():
            if isinstance(value, list):
                if max_param_values is None:
                    max_param_values = len(value)
                else:
                    assert max_param_values == len(value), "Params must be scalars or lists of equal length"
        return max_param_values

    def _create_folder(self, experiment_name) -> Path:
        # Todo: automatic numbering
        experiment_dir = config.EXPERIMENT_DIR / experiment_name
        try:
            os.makedirs(experiment_dir, exist_ok=False)
        except FileExistsError as e:
            raise e(f"Experiment named {experiment_name} already exists.")
        return experiment_dir


if __name__ == '__main__':
    exp = Experiment(
        n_layers=[1, 2, 1],
        hidden_size=148,
        discount_factor=0.5
    )
    exp.launch_experiment()