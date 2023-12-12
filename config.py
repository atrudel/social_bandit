import torch

POINTS_PER_TURN = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'