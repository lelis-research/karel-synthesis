import logging
import sys
import torch
from torch.utils.data import DataLoader
from dsl.production import Production
from embedding.autoencoder.program_vae import ProgramVAE
from embedding.config.config import Config
from embedding.models.TrainingModel import TrainingModel
from embedding.program_dataset import make_datasets


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dsl = Production.default_karel_production()

    config = Config(hidden_size=256)

    model = ProgramVAE(dsl, device, config)

    os.makedirs('output/logs', exist_ok=True)

    log_handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler('output/logs/stdout.txt', mode='w')] # TODO: include timestamp
    logging.basicConfig(handlers=log_handlers, format='%(asctime)s: %(message)s', level=logging.DEBUG)
    logger = logging.getLogger('trainer')

    p_train_dataset, p_val_dataset, p_test_dataset = make_datasets(
        'data/program_dataset', config.max_program_len, config.max_demo_length, 
        model.num_program_tokens, len(dsl.get_actions()) + 1, device, logger)

    p_train_dataloader = DataLoader(p_train_dataset, batch_size=256, shuffle=True, drop_last=True)
    p_val_dataloader = DataLoader(p_val_dataset, batch_size=256, shuffle=True, drop_last=True)
    p_test_dataloader = DataLoader(p_test_dataset, batch_size=256, shuffle=True, drop_last=True)

    train_model = TrainingModel(device, model, logger)

    train_model.train(p_train_dataloader, p_val_dataloader, 5)
