# Technical configuration
import numpy as np

DEVICE = 'cpu'
DATA_DIR = 'data'
MODEL_DIR = 'lightning_logs'

# Task-related configuration
POINTS_PER_TURN = 100

SEQUENCE_LENGTH = 80

TAU_FLUC = 3    # Fluctuation temperature for the latent means of the bandits
TAU_SAMP = 2    # Sampling temperature for actual bandit values
EPIMIN = 8      # Minimum length of an episode
EPIMAX = 32     # Maximum length of an episode
NEPI = 5        # How many episodes per trajectory


# Variables for the uncertainty generalization evaluation
GENERALIZATION_TAU_FLUCS = np.linspace(1, 5, num=8)
GENERALIZATION_TAU_SAMPS = np.linspace(0, 4, num=8)
GENERALIZATION_SET_SIZE = 100
