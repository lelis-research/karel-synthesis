import torch
from config import Config
from dsl import DSL
from logger.stdout_logger import StdoutLogger
from vae.models import load_model
from vae.program_dataset import make_dataloaders
from vae.trainer_translator import TrainerTranslator

if __name__ == '__main__':

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dsl = DSL.init_default_karel()

    model = load_model('DoubleVAETranslator', dsl, device)

    p_train_dataloader, p_val_dataloader, _ = make_dataloaders(dsl, device)

    trainer = TrainerTranslator(model)

    StdoutLogger.log('Main', f'Starting trainer for model {Config.model_name}')

    trainer.train(p_train_dataloader, p_val_dataloader)
    
    StdoutLogger.log('Main', 'Trainer finished.')
