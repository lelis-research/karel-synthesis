import torch.optim as optim
import torch.nn as nn
from experiment.data_loader import ProgramLoader
from experiment.model import StatePredictor


if __name__ == '__main__':

    n_epoch = 50
    data_loader = ProgramLoader('data/experiment_backup.hdf5')

    model = StatePredictor()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epoch):

        for i, data in enumerate(data_loader):

            z, s_s, s_f = data

            optimizer.zero_grad()

            output = model(s_s, z)
            loss = criterion(output, s_f)
            loss.backward()
            optimizer.step()
