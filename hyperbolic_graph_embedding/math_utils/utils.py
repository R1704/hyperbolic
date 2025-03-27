import torch

def g(x):
    return 2 * torch.exp(-x) / (1 + torch.exp(-x))