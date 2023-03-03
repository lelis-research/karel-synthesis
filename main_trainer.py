from datetime import datetime
import logging
import sys
import os
import torch
from torch.utils.data import DataLoader
from dsl.production import Production
from embedding.autoencoder.leaps_vae import LeapsVAE
from embedding.config.config import Config
from embedding.program_dataset import make_datasets
from embedding.trainer import Trainer


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dsl = Production.default_karel_production()

    config = Config()

    model = LeapsVAE(dsl, device, config)

    os.makedirs('output/logs', exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'output/logs/stdout_{timestamp}.txt', mode='w')
    ]
    logging.basicConfig(handlers=log_handlers, format='%(asctime)s: %(message)s', level=logging.DEBUG)
    logger = logging.getLogger('trainer')

    p_train_dataset, p_val_dataset, p_test_dataset = make_datasets(
        'data/program_dataset', config.max_program_len, config.max_demo_length, 
        model.num_program_tokens, model.num_agent_actions, device, logger)

    p_train_dataloader = DataLoader(p_train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    p_val_dataloader = DataLoader(p_val_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    p_test_dataloader = DataLoader(p_test_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    trainer = Trainer(model, 'output/leaps_vae', logger)

    trainer.train(p_train_dataloader, p_val_dataloader, 150)
