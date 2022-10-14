import numpy as np
import torch

# TODO: prever aleatoriamente onde não é parede (para ter ideia de como vai ser a loss)
# Também prever onde o Karel está (nenhum "movimento")

def predict_randomly(s_s: torch.Tensor):
    walls = s_s[:,:,:,4]
    preds = []
    for batch in walls:
        prediction = np.random.randint(64)
        x_pred = prediction // 8
        y_pred = prediction % 8
        while batch[x_pred, y_pred] != 0:
            prediction = np.random.randint(64)
            x_pred = prediction // 8
            y_pred = prediction % 8
        preds.append(prediction)
    return torch.Tensor(preds)

def predict_starting_position(s_s: torch.Tensor):
    pos = torch.moveaxis(s_s, [-1,-2,-3], [-3,-2,-1])
    pos = torch.sum(pos[:,:,:,0:4], dim=3)
    pos = torch.flatten(pos, start_dim=1)
    pos = torch.argmax(pos, dim=1)
    return pos