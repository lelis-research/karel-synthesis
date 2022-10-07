import numpy as np
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, '.')

from experiment.baseline import predict_starting_position
from experiment.data_loader import ProgramDataset
from experiment.model import StatePredictor

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
BATCH_SIZE = 256
RANDOM_SEED = 42

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = ProgramDataset('data/experiment_backup.hdf5')

    # Train / Val / Test split
    train_len = int(TRAIN_SPLIT * len(dataset))
    val_len = int(VAL_SPLIT * len(dataset))
    test_len = len(dataset) - train_len - val_len
    _, _, test_ds = random_split(dataset, 
        [train_len, val_len, test_len],
        torch.Generator().manual_seed(RANDOM_SEED)
    )

    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = StatePredictor().to(device)
    total = 0
    baseline_correct_predictions = 0

    for i, data in enumerate(test_dl):

        z, s_s, s_f = (d.to(device) for d in data)

        baseline_prediction = predict_starting_position(s_s)

        _, predicted = torch.max(baseline_prediction.data, 1)
        _, label = torch.max(s_f, 1)
        total += s_f.size(0)
        baseline_correct_predictions += (predicted == label).sum().item()

        # TODO: accuracy of test, print image of state

        # TODO: check why this is getting an accuracy of 0

    print(baseline_correct_predictions)
    print(total)
    print(baseline_correct_predictions / total)
