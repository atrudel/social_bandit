# Technical configuration
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
