import torch
from config.config import Config
from dsl.production import Production
from vae.models import load_model
from vae.program_dataset import make_dataloaders
from vae.trainer import Trainer

def main():

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dsl = Production.default_karel_production()

    model = load_model(Config.model_name, dsl, device)

    p_train_dataloader, p_val_dataloader, _ = make_dataloaders(dsl, device)

    trainer = Trainer(model)

    trainer.train(p_train_dataloader, p_val_dataloader)


if __name__ == '__main__':
    
    main()
