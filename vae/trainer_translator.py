import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from config import Config
from typing import NamedTuple
from logger.stdout_logger import StdoutLogger

from .models.base_vae import BaseVAE


class EpochReturn(NamedTuple):
    mean_total_loss: float
    mean_progs_t_accuracy: float
    mean_progs_s_accuracy: float


class TrainerTranslator:
    
    def __init__(self, model: BaseVAE):
        self.model = model
        self.output_dir = os.path.join('output', Config.experiment_name)
        self.disable_prog_teacher_enforcing = Config.trainer_disable_prog_teacher_enforcing
        self.disable_a_h_teacher_enforcing = Config.trainer_disable_a_h_teacher_enforcing
        self.save_each_epoch = Config.trainer_save_params_each_epoch
        self.num_epochs = Config.trainer_num_epochs
        self.device = self.model.device
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=Config.trainer_optim_lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=.95
        )
        self.loss_fn = nn.MSELoss(reduction='mean')
        
        os.makedirs(os.path.join(self.output_dir, 'model'), exist_ok=True)
        
    def _run_batch(self, batch: list, training = True) -> list:
        if training:
            self.model.train()
            torch.set_grad_enabled(True) # prob not needed
        else:
            self.model.eval()
            torch.set_grad_enabled(False) # prob not needed
            
        s_h, a_h, a_h_masks, progs, progs_masks = batch
        
        output = self.model(s_h, a_h, a_h_masks, progs, progs_masks,
                            not self.disable_prog_teacher_enforcing,
                            not self.disable_a_h_teacher_enforcing)
        z_prog_pred, z_prog, pred_progs, pred_progs_masks = output
        
        progs = progs[:, 1:].contiguous()
        progs_masks = progs_masks[:, 1:].contiguous()
        
        batch_size, num_demo_per_program, _ = a_h.shape
        
        z_prog_repeated = z_prog.repeat(num_demo_per_program, 1)
        progs_repeated = progs.repeat(num_demo_per_program, 1)
        progs_masks_repeated = progs_masks.repeat(num_demo_per_program, 1)
        
        if training:
            self.optimizer.zero_grad()
        
        loss = self.loss_fn(z_prog_pred, z_prog_repeated)
        
        if training:
            loss.backward()
            self.optimizer.step()
            
        with torch.no_grad():
            progs_masks_combined = torch.max(progs_masks_repeated, pred_progs_masks)
            progs_t_accuracy = (pred_progs[progs_masks_combined] == progs_repeated[progs_masks_combined]).float().mean()
            progs_s_accuracy = (progs_repeated == pred_progs).min(dim=1).values.float().mean()

        return [
            loss.detach().cpu().numpy().item(),
            progs_t_accuracy.detach().cpu().numpy().item(),
            progs_s_accuracy.detach().cpu().numpy().item()
        ]

    def _run_epoch(self, dataloader: DataLoader, epoch: int, training = True) -> EpochReturn:
        batch_info_list = np.zeros((len(dataloader), 3))
        
        for batch_idx, batch in enumerate(dataloader):
            batch_info = self._run_batch(batch, training)
            batch_info_list[batch_idx] = batch_info
        
        epoch_info_list = np.mean(batch_info_list, axis=0)
        
        return EpochReturn(*epoch_info_list.tolist())
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
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
        
        for epoch in range(1, self.num_epochs + 1):
            StdoutLogger.log('Trainer', f'Training epoch {epoch}.')
            train_info = self._run_epoch(train_dataloader, epoch, True)
            StdoutLogger.log('Trainer', train_info._asdict())
            with open(os.path.join(self.output_dir, 'training_info.csv'), mode='a') as f:
                f.write(f"{epoch},")
                f.write(",".join([str(i) for i in train_info]))
                f.write("\n")
            if self.save_each_epoch:
                parameters_path = os.path.join(self.output_dir, 'model', f'epoch_{epoch}.ptp')
                torch.save(self.model.state_dict(), parameters_path)
                StdoutLogger.log('Trainer', f'Parameters saved in {parameters_path}')
 
            if val_dataloader is not None:
                StdoutLogger.log('Trainer',f'Validation epoch {epoch}.')
                val_info = self._run_epoch(val_dataloader, epoch, False)
                StdoutLogger.log('Trainer',val_info._asdict())
                with open(os.path.join(self.output_dir, 'validation_info.csv'), mode='a') as f:
                    f.write(f"{epoch},")
                    f.write(",".join([str(i) for i in val_info]))
                    f.write("\n")
                val_return = val_info._asdict()[validation_key]
 
                if val_return < best_val_return:
                    best_val_return = val_return
                    StdoutLogger.log('Trainer',f'New best validation {validation_key}: {best_val_return}')
                    parameters_path = os.path.join(self.output_dir, 'model', 'best_val.ptp')
                    torch.save(self.model.state_dict(), parameters_path)
                    StdoutLogger.log('Trainer',f'Parameters saved in {parameters_path}')
