import logging
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from embedding.autoencoder.base_vae import BaseVAE
from typing import NamedTuple


class EpochReturn(NamedTuple):
    mean_total_loss: float
    mean_progs_loss: float
    mean_a_h_loss: float
    mean_latent_loss: float
    mean_progs_t_accuracy: float
    mean_progs_s_accuracy: float
    mean_a_h_t_accuracy: float
    mean_a_h_s_accuracy: float


class Trainer:
    
    def __init__(self, model: BaseVAE, output_dir: str, logger: logging.Logger,
                 prog_loss_coef = 1.0, a_h_loss_coef = 1.0, latent_loss_coef = 0.1,
                 optim_lr = 5e-4):
        self.model = model
        self.output_dir = output_dir
        self.logger = logger
        self.prog_loss_coef = prog_loss_coef
        self.a_h_loss_coef = a_h_loss_coef
        self.latent_loss_coef = latent_loss_coef
        self.device = self.model.device
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=optim_lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=.95
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
        os.makedirs(os.path.join(self.output_dir, 'model'), exist_ok=True)
        
    def _run_batch(self, batch: list, training = True) -> list:
        if training:
            self.model.train()
            torch.set_grad_enabled(True) # prob not needed
        else:
            self.model.eval()
            torch.set_grad_enabled(False) # prob not needed
            
        s_h, a_h, a_h_masks, progs, progs_masks = batch
        
        output = self.model(s_h, a_h, a_h_masks, progs, progs_masks)
        pred_progs, pred_progs_logits, pred_progs_masks,\
            pred_a_h, pred_a_h_logits, pred_a_h_masks = output
        
        # Combine first 2 dimensions of a_h (batch_size and demos_per_program)
        a_h = a_h.view(-1, a_h.shape[-1])
        a_h_masks = a_h_masks.view(-1, a_h.shape[-1])
        
        # Skip first token in ground truth sequences
        progs = progs[:, 1:].contiguous()
        progs_masks = progs_masks[:, 1:].contiguous()
        a_h = a_h[:, 1:].contiguous()
        a_h_masks = a_h_masks[:, 1:].contiguous()

        # Flatten everything for loss calculation
        progs_flat = progs.view(-1, 1)
        progs_masks_flat = progs_masks.view(-1, 1)
        a_h_flat = a_h.view(-1, 1)
        a_h_masks_flat = a_h_masks.view(-1, 1)
        
        pred_progs_logits = pred_progs_logits.view(-1, pred_progs_logits.shape[-1])
        pred_a_h_logits = pred_a_h_logits.view(-1, pred_a_h_logits.shape[-1])
        
        pred_progs_masks_flat = pred_progs_masks.view(-1, 1)
        pred_a_h_masks_flat = pred_a_h_masks.view(-1, 1)
        
        # We combine masks here to penalize predictions that are larger than ground truth
        progs_masks_flat_combined = torch.max(progs_masks_flat, pred_progs_masks_flat).squeeze()
        a_h_masks_flat_combined = torch.max(a_h_masks_flat, pred_a_h_masks_flat).squeeze()
        
        if training:
            self.optimizer.zero_grad()
        
        # Calculate classification loss only on tokens in mask
        progs_loss = self.loss_fn(pred_progs_logits[progs_masks_flat_combined],
                                  progs_flat[progs_masks_flat_combined].view(-1))
        a_h_loss = self.loss_fn(pred_a_h_logits[a_h_masks_flat_combined],
                                a_h_flat[a_h_masks_flat_combined].view(-1))
        
        latent_loss = self.model.get_latent_loss()
        
        total_loss = self.prog_loss_coef * progs_loss + self.a_h_loss_coef * a_h_loss \
            + self.latent_loss_coef * latent_loss
            
        if training:
            total_loss.backward()
            self.optimizer.step()
            
        with torch.no_grad():
            progs_masks_combined = torch.max(progs_masks, pred_progs_masks)
            progs_t_accuracy = (pred_progs[progs_masks_combined] == progs[progs_masks_combined]).float().mean()
            progs_s_accuracy = (progs == pred_progs).min(dim=1).values.float().mean()
            
            a_h_masks_combined = torch.max(a_h_masks, pred_a_h_masks)
            a_h_t_accuracy = (pred_a_h[a_h_masks_combined] == a_h[a_h_masks_combined]).float().mean()
            a_h_s_accuracy = (a_h == pred_a_h).min(dim=1).values.float().mean()
            
        return [
            total_loss.detach().cpu().numpy().item(),
            progs_loss.detach().cpu().numpy().item(),
            a_h_loss.detach().cpu().numpy().item(),
            latent_loss.detach().cpu().numpy().item(),
            progs_t_accuracy.detach().cpu().numpy().item(),
            progs_s_accuracy.detach().cpu().numpy().item(),
            a_h_t_accuracy.detach().cpu().numpy().item(),
            a_h_s_accuracy.detach().cpu().numpy().item()
        ]

    def _run_epoch(self, dataloader: DataLoader, epoch: int, training = True) -> EpochReturn:
        batch_info_list = np.zeros((len(dataloader), 8))
        
        for batch_idx, batch in enumerate(dataloader):
            batch_info = self._run_batch(batch, training)
            batch_info_list[batch_idx] = batch_info
        
        epoch_info_list = np.mean(batch_info_list, axis=0)
        
        return EpochReturn(*epoch_info_list.tolist())
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, max_epoch = 150):
        if val_dataloader is not None:
            validation_key = 'mean_total_loss'
            best_val_return = np.inf
        
        with open(os.path.join(self.output_dir, 'training_info.csv'), mode='w') as f:
            f.write("epoch,")
            f.write(",".join(EpochReturn._fields))
            f.write("\n")

        if val_dataloader is not None:
            with open(os.path.join(self.output_dir, 'validation_info.csv'), mode='w') as f:
                f.write("epoch,")
                f.write(",".join(EpochReturn._fields))
                f.write("\n")
        
        for epoch in range(1, max_epoch+1):
            self.logger.info(f'Training epoch {epoch}.')
            train_info = self._run_epoch(train_dataloader, epoch, True)
            self.logger.info(train_info._asdict())
            with open(os.path.join(self.output_dir, 'training_info.csv'), mode='a') as f:
                f.write(f"{epoch},")
                f.write(",".join([str(i) for i in train_info]))
                f.write("\n")
            parameters_path = os.path.join(self.output_dir, 'model', f'epoch_{epoch}.ptp')
            torch.save(self.model.state_dict(), parameters_path)
            self.logger.info(f'Parameters saved in {parameters_path}')
 
            if val_dataloader is not None:
                self.logger.info(f'Validation epoch {epoch}.')
                val_info = self._run_epoch(val_dataloader, epoch, False)
                self.logger.info(val_info._asdict())
                with open(os.path.join(self.output_dir, 'validation_info.csv'), mode='a') as f:
                    f.write(f"{epoch},")
                    f.write(",".join([str(i) for i in val_info]))
                    f.write("\n")
                val_return = val_info._asdict()[validation_key]
 
                if val_return < best_val_return:
                    best_val_return = val_return
                    self.logger.info(f'New best validation {validation_key}: {best_val_return}')
                    parameters_path = os.path.join(self.output_dir, 'model', 'best_val.ptp')
                    torch.save(self.model.state_dict(), parameters_path)
                    self.logger.info(f'Parameters saved in {parameters_path}')
                    
                    
