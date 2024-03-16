import argparse

from social_bandit.training.experiment import Experiment
from social_bandit.training.objective_functions import AdvantageObjFunc, EntropyObjFunc
from social_bandit.data_generation.data_generator import GeneralizationDatasetBundle


parser = argparse.ArgumentParser(description="Launching an experiment to train a model on Social Bandit")
parser.add_argument('--debug', action='store_true', help='debug mode')
args = parser.parse_args()

exp = Experiment(
    n_layers=1,
    hidden_size=48,
    lr=1e-3,
    obj_func=[
        AdvantageObjFunc(),
        EntropyObjFunc(AdvantageObjFunc(), 0.5),
        EntropyObjFunc(AdvantageObjFunc(), 1),
    ],
    n_epochs=1,
)
exp.launch_experiment(
    "Debug",
    seeds=1,
    debug=args.debug
)
