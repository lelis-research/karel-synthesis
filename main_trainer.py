import torch
from config.config import Config
from dsl.production import Production
from logger.stdout_logger import StdoutLogger
from vae.models import load_model
from vae.program_dataset import make_dataloaders
from vae.trainer import Trainer

if __name__ == '__main__':

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dsl = Production.default_karel_production()

    model = load_model(Config.model_name, dsl, device)

    p_train_dataloader, p_val_dataloader, _ = make_dataloaders(dsl, device)

    trainer = Trainer(model)

    StdoutLogger.log('Main', f'Starting trainer for model {Config.model_name}')

    trainer.train(p_train_dataloader, p_val_dataloader)
    
    StdoutLogger.log('Main', 'Trainer finished.')
