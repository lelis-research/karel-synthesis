import torch
from dsl.production import Production
from vae.models.sketch_vae import SketchVAE
from vae.program_dataset import SketchDataset, make_dataloaders
from vae.trainer import Trainer

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dsl = Production.default_karel_production(include_hole=True)

    model = SketchVAE(dsl, device)

    p_train_dataloader, p_val_dataloader, _ = make_dataloaders(dsl, device, SketchDataset)

    trainer = Trainer(model)

    trainer.train(p_train_dataloader, p_val_dataloader)


if __name__ == '__main__':
    
    main()
