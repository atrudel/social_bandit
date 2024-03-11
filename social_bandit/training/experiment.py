import glob
import os
from pathlib import Path
from typing import List
import re

from social_bandit import config
from social_bandit.game.chooser import Chooser
from social_bandit.game.environment import Env
from social_bandit.game.partner import DataPartner
from social_bandit.models.rnn_chooser import RNNChooserPolicy
from social_bandit.training.objective_functions import RewardObjectiveFunction
from social_bandit.training.trainer import Trainer


class Experiment:
    def __init__(self, **params):
        self.training_params = self._pack_training_params(params)

    def launch_experiment(self, name: str, debug: bool = False):
        self.name: str = name
        if not debug:
            experiment_dir: Path = self._create_folder(name)

        for i, params in enumerate(self.training_params):
            chooser_policy_model = RNNChooserPolicy(
                hidden_size=params['hidden_size'],
                n_layers=params['n_layers']
            )
            objective_function = RewardObjectiveFunction(discount_factor=params['discount_factor'])
            env = Env(
                chooser=Chooser(chooser_policy_model),
                partner_0=DataPartner(),
                partner_1=DataPartner()
            )
            trainer = Trainer(
                env=env,
                trained_model=chooser_policy_model,
                objective_function=objective_function,
                training_name=f"train{i:02}",
                experiment_dir=experiment_dir,
                data_dir=config.DATA_DIR,
            )
            trainer.launch_training(
                n_epochs=params['n_epochs'],
                seed=params['seed'],
                learning_rate=params['learning_rate'],
                batch_size=config.BATCH_SIZE,
                debug=debug
            )
            if not debug:
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
        exp_number: int = self._find_next_experiment_number()
        experiment_dir = config.EXPERIMENT_DIR / f"exp{exp_number:02d}_{experiment_name}"
        os.makedirs(experiment_dir, exist_ok=False)
        return experiment_dir

    def _find_next_experiment_number(self) -> int:
        other_dir_names: List[str] = glob.glob(f"exp*", root_dir=config.EXPERIMENT_DIR)
        other_dir_names = sorted(other_dir_names)
        try:
            last_dir_name: str = other_dir_names[-1]
        except IndexError:
            return 0
        match = re.match(r"exp(\d{2}).*", last_dir_name)
        last_exp_number: str = match.group(1)
        new_exp_number = int(last_exp_number) + 1
        return new_exp_number


if __name__ == '__main__':
    exp = Experiment(
        n_layers=2,
        hidden_size=148,
        seed=[1, 2, 3, 4],
        learning_rate=1e-3,
        discount_factor=0.5,
        n_epochs=5,

    )
    exp.launch_experiment("RL_convergence")