import numpy as np
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import tqdm

sys.path.insert(0, '.')

from experiment.data_loader import ProgramDataset
from experiment.model import StatePredictor

N_EPOCH = 50
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
BATCH_SIZE = 1024
RANDOM_SEED = 42

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = ProgramDataset('data/experiment.hdf5')

    # Train / Val / Test split
    train_len = int(TRAIN_SPLIT * len(dataset))
    val_len = int(VAL_SPLIT * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_ds, val_ds, test_ds = random_split(dataset, 
        [train_len, val_len, test_len],
        torch.Generator().manual_seed(RANDOM_SEED)
    )

    # Data loaders
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = StatePredictor().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Training
    min_val_loss = np.inf
    for epoch in range(N_EPOCH):

        model.train()
        train_loss = 0
        train_accuracy = 0

        with tqdm.tqdm(train_dl, unit='batch', desc=f'Epoch {epoch} Training') as train_batch:

            for i, data in enumerate(train_batch):

                z, s_s, s_f = (d.to(device) for d in data)

                optimizer.zero_grad()

                output = model(s_s, z)
                loss = criterion(output, s_f)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                train_batch.set_postfix(loss=loss.item())

                train_accuracy += (output == s_f)
        
        print(f'Epoch {epoch} training loss: {train_loss / len(train_dl)}')

        model.eval()
        val_loss = 0.0

        with tqdm.tqdm(val_dl, unit='batch', desc=f'Epoch {epoch} Validation') as val_batch:

            for i, data in enumerate(val_batch):

                z, s_s, s_f = (d.to(device) for d in data)

                # TODO: accuracy?

                output = model(s_s, z)
                loss = criterion(output, s_f)

                val_loss += loss.item()

                val_batch.set_postfix(loss=loss.item())
        
        print(f'Epoch {epoch} validation loss: {val_loss / len(val_dl)}')

        if min_val_loss > val_loss:

            min_val_loss = val_loss

            # Save parameters if new best validation
            torch.save(model.state_dict(), 'data/experiment_model.pth')
