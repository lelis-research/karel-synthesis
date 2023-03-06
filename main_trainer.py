from argparse import ArgumentParser
from datetime import datetime
import logging
import sys
import os
import torch
from dsl.production import Production
from embedding.autoencoder.leaps_vae import LeapsVAE
from config.config import Config
from embedding.program_dataset import make_dataloaders
from embedding.trainer import Trainer

def main(config: Config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dsl = Production.default_karel_production()

    model = LeapsVAE(dsl, device, config)
    
    output_folder = os.path.join('output', model.name)

    os.makedirs(os.path.join(output_folder, 'logs'), exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(output_folder, 'logs', f'{timestamp}.txt'), mode='w')
    ]
    logging.basicConfig(handlers=log_handlers, format='%(asctime)s: %(message)s', level=logging.DEBUG)
    logger = logging.getLogger('trainer')

    p_train_dataloader, p_val_dataloader, _ = make_dataloaders('data/program_dataset', dsl, device, logger, config)

    trainer = Trainer(model, logger, config)

    trainer.train(p_train_dataloader, p_val_dataloader)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='program_vae')
    parser.add_argument('--model_hidden_size', type=int, default=256)
    parser.add_argument('--data_batch_size', type=int, default=256)
    parser.add_argument('--data_max_program_len', type=int ,default=45)
    parser.add_argument('--data_max_demo_length', type=int, default=100)
    parser.add_argument('--data_num_demo_per_program', type=int, default=10)
    parser.add_argument('--env_height', type=int, default=8)
    parser.add_argument('--env_width', type=int, default=8)
    parser.add_argument('--trainer_num_epochs', type=int, default=150)
    parser.add_argument('--trainer_prog_teacher_enforcing', type=bool, default=True)
    parser.add_argument('--trainer_a_h_teacher_enforcing', type=bool, default=True)
    args = parser.parse_args()

    config = Config(**vars(args))
    
    main(config)
