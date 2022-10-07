import numpy as np
import torch

# TODO: prever aleatoriamente onde não é parede (para ter ideia de como vai ser a loss)
# Também prever onde o Karel está (nenhum "movimento")

def predict_randomly(s_s: torch.Tensor):
    walls = s_s[:,:,4]
    

def predict_starting_position(s_s: torch.Tensor):
    
    pos = torch.sum(s_s[:,:,:,0:4], dim=2)
    return torch.flatten(pos, start_dim=1)