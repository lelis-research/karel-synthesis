import torch
from dsl.production import Production
from embedding.autoencoder.leaps_vae import LeapsVAE
from config.config import Config
from embedding.program_dataset import make_dataloaders
from embedding.trainer import Trainer
from logger.stdout_logger import StdoutLogger

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dsl = Production.default_karel_production()

    model = LeapsVAE(dsl, device)
    
    StdoutLogger.init_logger('trainer')

    p_train_dataloader, p_val_dataloader, _ = make_dataloaders('data/program_dataset', dsl, device)

    trainer = Trainer(model)

    trainer.train(p_train_dataloader, p_val_dataloader)


if __name__ == '__main__':

    Config.parse_args()
    
    main()
