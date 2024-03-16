from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Union

from social_bandit import config
from social_bandit.evaluation.chooser_evaluator import ChooserEvaluator
from social_bandit.game.chooser import Chooser
from social_bandit.game.environment import Env
from social_bandit.game.partner import DataPartner
from social_bandit.models.rnn_chooser import RNNChooserPolicy
from social_bandit.training.trainer import Trainer


class Experiment:
    def __init__(self, **params):
        self.training_params: List[dict] = self._pack_training_params(params)
        self.variable_params: List[str] = self._get_variable_params(params)
        self.trainings: List[Trainer] = []
        self.experiment_dir: Optional[Path] = None
        self.name = None

    def launch_experiment(self, name: str, seeds: Union[int, List[int]], debug: bool = False) -> ChooserEvaluator:
        self.name: str = name
        seeds: List[int] = [seeds] if isinstance(seeds, int) else seeds

        if not debug:
            # Link git code versioning to experiment number
            experiment_number: int = self._find_experiment_number()
            self.tag_experiment_number_in_git(experiment_number)

            # Save experiment description to folder
            self.experiment_dir: Path = self._create_folder(experiment_number, name)
            with open(self.experiment_dir / 'description.txt', 'w') as f:
                pprint(f'Experiment: {name}', stream=f)
                pprint(f"Training Params:", stream=f)
                pprint(self.training_params, stream=f)

        # Separate training for each parameter configuration
        for i, params in enumerate(self.training_params):
            # Duplicate trainings for each seed provided
            for j, seed in enumerate(seeds):
                training_description = ", ".join([f"{param_name}={str(params[param_name])},"
                                                  for param_name in self.variable_params])
                chooser_policy_model = RNNChooserPolicy(
                    hidden_size=params['hidden_size'],
                    n_layers=params['n_layers']
                )
                chooser = Chooser(chooser_policy_model, training_description)
                env = Env(
                    chooser=chooser,
                    partner_0=DataPartner(),
                    partner_1=DataPartner()
                )
                objective_function = params['obj_func']
                trainer = Trainer(
                    env=env,
                    trained_model=chooser_policy_model,
                    obj_func=objective_function,
                    training_name=f"train{i:02}.{j}",
                    experiment_dir=self.experiment_dir,
                    data_dir=config.DATA_DIR,
                )
                trainer.launch_training(
                    n_epochs=params['n_epochs'],
                    seed=seed,
                    learning_rate=params['lr'],
                    batch_size=config.BATCH_SIZE,
                    debug=debug
                )
                self.trainings.append(trainer)
                if not debug:
                    trainer.save_agents()

        evaluator: ChooserEvaluator = self.evaluate_chooser(debug=debug)
        return evaluator

    def evaluate_chooser(self, debug=False) -> ChooserEvaluator:
        if debug:
            save_dir = None
        else:
            save_dir = self.experiment_dir / 'evaluation_chooser'
            os.makedirs(save_dir)
        envs = [training.env for training in self.trainings]
        evaluator = ChooserEvaluator(envs, save_dir=save_dir)
        evaluator.run_evaluations(seed=0, show=debug)
        return evaluator

    def _get_variable_params(self, params) -> List[str]:
        variable_params = []
        for key, value in params.items():
            if isinstance(value, list):
                variable_params.append(key)
        return variable_params

    def _pack_training_params(self, params) -> List[dict]:
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

    def _count_number_of_trainings(self, params) -> int:
        max_param_values = None
        for value in params.values():
            if isinstance(value, list):
                if max_param_values is None:
                    max_param_values = len(value)
                else:
                    assert max_param_values == len(value), "Params must be scalars or lists of equal length"
        return max_param_values

    @staticmethod
    def _create_folder(exp_number, exp_name) -> Path:
        experiment_dir = config.EXPERIMENT_DIR / f"exp{exp_number:03d}_{exp_name}"
        os.makedirs(experiment_dir, exist_ok=False)
        return experiment_dir

    @staticmethod
    def _find_experiment_number() -> int:
        other_dir_names: List[str] = glob.glob(f"exp*", root_dir=config.EXPERIMENT_DIR)
        other_dir_names = sorted(other_dir_names)
        try:
            last_dir_name: str = other_dir_names[-1]
        except IndexError:
            return 0
        match = re.match(r"exp(\d{3}).*", last_dir_name)
        last_exp_number: str = match.group(1)
        new_exp_number = int(last_exp_number) + 1
        return new_exp_number

    @staticmethod
    def tag_experiment_number_in_git(exp_number: int):
        # Check the current git status
        git_status_output = subprocess.check_output(
            "git status --porcelain --untracked-files=no",
            shell=True, text=True
        )
        # If git working tree is not clean (uncommitted changes), exit the program
        if git_status_output.strip():
            print("\033[31mCOMMIT all your changes before you run the training script.\033[0m")
            sys.exit(1)

        # Get the short commit hash
        commit_hash = subprocess.check_output(
            "git rev-parse --short=7 HEAD",
            shell=True,
            text=True
        ).strip()
        tag_name = f"exp{exp_number:03d}"
        try:
            subprocess.run(["git", "tag", tag_name], check=True)
            print(f"Creating git tag: <{tag_name}> for commit {commit_hash}")
        except subprocess.CalledProcessError as e:
            print(f"Unable to create git tag with version name: {e}")
