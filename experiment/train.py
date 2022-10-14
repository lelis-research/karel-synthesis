import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data.data_loader import ProgramDataset
from model.predictor import StatePredictor

N_EPOCH = 250
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
BATCH_SIZE = 1024
RANDOM_SEED = 42

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = ProgramDataset('data/experiment.npz')

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
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    max_val_accuracy = 0
    for epoch in range(N_EPOCH):

        model.train()
        train_loss = 0.
        train_accuracy = 0.
        train_total = 0

        print(f'Epoch {epoch}: Starting training...')

        for i, data in enumerate(train_dl):

            z, s_s, s_f, _ = (d.to(device) for d in data)

            optimizer.zero_grad()

            output = model(s_s, z)
            loss = criterion(output, s_f)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            predicton_output = torch.argmax(output, dim=1)
            train_accuracy += (predicton_output == s_f).float().sum()

            train_total += z.size(0)
        
        print(f'Epoch {epoch} training loss: {train_loss / train_total}')
        print(f'Epoch {epoch} training acc: {train_accuracy / train_total}')

        train_losses.append(train_loss / train_total)
        train_accs.append(train_accuracy / train_total)

        model.eval()
        val_loss = 0.
        val_accuracy = 0.
        val_total = 0

        for i, data in enumerate(val_dl):

            z, s_s, s_f = (d.to(device) for d in data)

            output = model(s_s, z)
            loss = criterion(output, s_f)

            val_loss += loss.item()

            predicton_output = torch.argmax(output, dim=1)
            val_accuracy += (predicton_output == s_f).float().sum()

            val_total += z.size(0)
    
        print(f'Epoch {epoch} validation loss: {val_loss / val_total}')
        print(f'Epoch {epoch} validation acc: {val_accuracy / val_total}')

        val_losses.append(val_loss / val_total)
        val_accs.append(val_accuracy / val_total)

        if max_val_accuracy < val_accuracy:

            max_val_accuracy = val_accuracy

            print('New best validation accuracy, saving model...')

            # Save parameters if new best validation
            torch.save(model.state_dict(), 'output/experiment_model.pth')

    with open('output/experiment_training_data.csv', mode='w') as f:
        f.write('train_loss,train_acc,val_loss,val_acc')
        for train_loss, train_acc, val_loss, val_acc in zip(
            train_losses, train_accs, val_losses, val_accs
        ):
            f.write(f'\n{train_loss},{train_acc},{val_loss},{val_acc}')