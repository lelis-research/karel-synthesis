from argparse import ArgumentParser
from datetime import datetime
import logging
import sys
import os
import torch
from dsl.production import Production
from embedding.autoencoder.policy_vae import PolicyVAE
from embedding.program_dataset import make_dataloaders
from embedding.trainer import Trainer

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dsl = Production.default_karel_production()

    model = PolicyVAE(dsl, device)

    p_train_dataloader, p_val_dataloader, _ = make_dataloaders('data/program_dataset', dsl, device)

    trainer = Trainer(model)

    trainer.train(p_train_dataloader, p_val_dataloader)


if __name__ == '__main__':
    
    main()
