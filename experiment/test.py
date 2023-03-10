import torch
from torch.utils.data import DataLoader, random_split
from model.baseline import predict_randomly, predict_starting_position
from data.data_loader import ProgramDataset
from model.predictor import StatePredictor

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
BATCH_SIZE = 256
RANDOM_SEED = 42

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = ProgramDataset('data/experiment.npz')

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
    test_total = 0.
    random_baseline_accuracy = 0.
    starting_baseline_accuracy = 0.
    test_accuracy = 0.

    len_k = 5
    topk_accuracy = [0. for i in range(len_k)]

    model.load_state_dict(torch.load('output/experiment_model.pth', map_location=device))

    for i, data in enumerate(test_dl):

        z, s_s, s_f, _ = (d.to(device) for d in data)

        output = model(s_s, z)

        prediction_output = torch.argmax(output, dim=1)
        test_accuracy += (prediction_output == s_f).float().sum()

        prediction_topk = torch.topk(output, k=len_k, dim=1).indices
        for i in range(len_k):
            topk_accuracy[i] += (prediction_topk[:,i] == s_f).float().sum()

        random_baseline = predict_randomly(s_s)
        random_baseline_accuracy += (random_baseline == s_f).sum().item()

        starting_baseline = predict_starting_position(s_s)
        starting_baseline_accuracy += (starting_baseline == s_f).sum().item()

        test_total += s_f.size(0)

    print(f'Test accuracy: {test_accuracy / test_total}')
    print(f'TopK accuracy: {[acc / test_total for acc in topk_accuracy]}')
    print(f'Random Baseline accuracy: {random_baseline_accuracy / test_total}')
    print(f'Starting position Baseline accuracy: {starting_baseline_accuracy / test_total}')
