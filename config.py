import os
from pathlib import Path

import numpy as np

root_directory = Path(os.path.dirname(os.path.realpath(__file__)))

# Model training configuration
DEVICE = 'cpu'
DATA_DIR = root_directory / 'data'
MODEL_DIR = 'lightning_logs' # Deprecated
EXPERIMENT_DIR = root_directory / 'experiments'
BATCH_SIZE = 1000
VALIDATE_EVERY_NSTEPS = 10

# Task-related configuration
POINTS_PER_TURN = 100
N_TRIALS = 80


# Data generation configuration
TAU_FLUC = 3    # Fluctuation temperature for the latent means of the bandits
TAU_SAMP = 2    # Sampling temperature for actual bandit values
EPIMIN = 8      # Minimum length of an episode
EPIMAX = 32     # Maximum length of an episode
NEPI = 5        # How many episodes per trajectory


# Variables for the uncertainty generalization evaluation
GENERALIZATION_TAU_FLUCS = np.linspace(1, 5, num=8)
GENERALIZATION_TAU_SAMPS = np.linspace(0, 4, num=8)
GENERALIZATION_SET_SIZE = 100
