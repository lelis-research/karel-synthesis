import torch
from dsl.production import Production
from embedding.autoencoder.sketch_vae import SketchVAE
from embedding.program_dataset import make_dataloaders
from embedding.trainer import Trainer

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dsl = Production.default_karel_production()

    model = SketchVAE(dsl, device)

    p_train_dataloader, p_val_dataloader, _ = make_dataloaders(dsl, device)

    trainer = Trainer(model)

    trainer.train(p_train_dataloader, p_val_dataloader)


if __name__ == '__main__':
    
    main()
