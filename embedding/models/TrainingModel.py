import logging
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from dsl.parser import Parser

from embedding.autoencoder.program_vae import ProgramVAE


class TrainingModel(object):

    def __init__(self, device: torch.device, net: ProgramVAE, latent_loss_coef = 1.0, condition_loss_coef = 1.0, optim_lr = 5e-4) -> None:
        self.device = device
        self.net = net
        self.latent_loss_coef = latent_loss_coef
        self.condition_loss_coef = condition_loss_coef
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=optim_lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=.95
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.epoch = 0

    def masked_mean(x, mask, dim=-1, keepdim=False):
        assert x.shape == mask.shape
        return torch.sum(x * mask.float(), dim=dim, keepdim=keepdim) / torch.sum(mask, dim=dim, keepdim=keepdim)

    def calculate_accuracy(self, logits, targets, mask, batch_shape):
        masked_preds = (logits.argmax(dim=-1, keepdim=True) * mask).view(*batch_shape, 1)
        masked_targets = (targets * mask).view(*batch_shape, 1)
        t_accuracy = 100 * self.masked_mean(
            (masked_preds == masked_targets).float(),
            mask.view(*masked_preds.shape), dim=1
        ).mean()

        p_accuracy = 100 * (masked_preds.squeeze() == masked_targets.squeeze()).all(dim=1).float().mean()
        return t_accuracy, p_accuracy

    def _get_condition_loss(self, a_h, a_h_len, action_logits, action_masks):
        """ loss between ground truth trajectories and predicted action sequences

        :param a_h(int16): B x num_demo_per_program x max_demo_length
        :param a_h_len(int16): a_h_len: B x num_demo_per_program
        :param action_logits: (B * num_demo_per_programs) x max_a_h_len x num_actions
        :param action_masks: (B * num_demo_per_programs) x max_a_h_len x 1
        :return (float): condition policy loss
        """
        batch_size_x_num_demo_per_program, max_a_h_len, num_actions = action_logits.shape
        assert max_a_h_len == a_h.shape[-1]

        padded_preds = action_logits

        """ add dummy logits to targets """
        target_masks = a_h != self.net.condition_policy.num_agent_actions - 1
        # remove temporarily added no-op actions in case of empty trajectory to
        # verify target masks
        a_h_len2 = a_h_len - (a_h[:,:,0] == self.net.condition_policy.num_agent_actions - 1).to(a_h_len.dtype)
        assert (target_masks.sum(dim=-1).squeeze() == a_h_len2.squeeze()).all()
        targets = torch.where(target_masks, a_h, (num_actions-1) * torch.ones_like(a_h))

        """ condition mask """
        # flatten everything and select actions that you want in backpropagation
        target_masks = target_masks.view(-1, 1)
        action_masks = action_masks.view(-1, 1)
        cond_mask = torch.max(action_masks, target_masks)

        # gather prediction that needs backpropagation
        subsampled_targets = targets.view(-1,1)[cond_mask].long()
        subsampled_padded_preds = padded_preds.view(-1, num_actions)[cond_mask.squeeze()]

        condition_loss = self.loss_fn(subsampled_padded_preds, subsampled_targets)

        """ calculate accuracy """
        with torch.no_grad():
            batch_shape = padded_preds.shape[:-1]
            cond_t_accuracy, cond_p_accuracy = self.calculate_accuracy(
                padded_preds.view(-1, num_actions),
                targets.view(-1, 1), cond_mask, batch_shape
            )

        return condition_loss, cond_t_accuracy, cond_p_accuracy

    def _run_batch(self, batch, training: bool) -> dict:
        if training:
            self.net.train()
            torch.set_grad_enabled(True)
        else:
            self.net.eval()
            torch.set_grad_enabled(False)

        programs, ids, trg_mask, s_h, a_h, a_h_len = batch

        # Forward pass
        output = self.net(programs, trg_mask, s_h, a_h, deterministic=True)
        pred_programs, pred_program_lens, output_logits, eop_pred_programs, eop_output_logits, pred_program_masks,\
        action_logits, action_masks, z = output

        # Skip first token DEF for loss calculation TODO: Why is this done?
        targets = programs[:, 1:].contiguous().view(-1, 1)
        trg_mask = trg_mask[:, 1:].contiguous().view(-1, 1)
        logits = output_logits.view(-1, output_logits.shape[-1])
        pred_mask = pred_program_masks.view(-1, 1)
        # need to penalize shorter and longer predicted programs
        vae_mask = torch.max(pred_mask, trg_mask)

        if training:
            self.optimizer.zero_grad()

        rec_loss = self.loss_fn(logits[vae_mask.squeeze()], (targets[vae_mask.squeeze()]).view(-1))
        lat_loss = self.net.vae.latent_loss(self.net.vae.z_mean, self.net.vae.z_sigma)
        condition_loss, cond_t_accuracy, cond_p_accuracy = self._get_condition_loss(a_h, a_h_len, action_logits,
                                                                                        action_masks)

        loss = rec_loss + (self.latent_loss_coef * lat_loss) + (self.condition_loss_coef * condition_loss)

        if training:
            loss.backward()
            self.optimizer.step()

        # Calculate accuracy
        with torch.no_grad():
            batch_shape = output_logits.shape[:-1]
            t_accuracy, p_accuracy = self.calculate_accuracy(logits, targets, vae_mask, batch_shape)

            if training:
                zero_tensor = torch.tensor([0.0], device=self.device, requires_grad=False)
                generated_programs, glogits = None, None
                greedy_t_accuracy, greedy_p_accuracy, greedy_a_accuracy, greedy_d_accuracy = zero_tensor, zero_tensor, zero_tensor, zero_tensor
            else:
                # greedy rollout of decoder
                greedy_outputs = self.net.vae.decoder(programs, z, teacher_enforcing=False, deterministic=True)
                _, _, _, _, greedy_output_logits, _, _, pred_program_masks, _ = greedy_outputs

                """ calculate accuracy """
                logits = greedy_output_logits.view(-1, greedy_output_logits.shape[-1])
                pred_mask = pred_program_masks.view(-1, 1)
                vae_mask = torch.max(pred_mask, trg_mask)
                batch_shape = greedy_output_logits.shape[:-1]
                greedy_t_accuracy, greedy_p_accuracy = self.calculate_accuracy(
                    logits, targets, vae_mask, batch_shape
                )

                _, _, _, action_logits, action_masks, _ = self.net.condition_policy(
                    s_h, a_h, z, teacher_enforcing=False, deterministic=True
                )
                _, greedy_a_accuracy, greedy_d_accuracy = self._get_condition_loss(
                    a_h, a_h_len, action_logits, action_masks
                )

                # 2 random vectors
                generated_programs = None
                rand_z = torch.randn((2, z.shape[1])).to(z.dtype).to(z.device)
                generated_outputs = self.net.vae.decoder(None, rand_z, teacher_enforcing=False, deterministic=True)
                generated_programs = [Parser.list_to_tokens(prg.detach().cpu().numpy().tolist()) for prg in generated_outputs[1]]

        return {
            'decoder_token_accuracy': t_accuracy.detach().cpu().numpy().item(),
            'decoder_program_accuracy': p_accuracy.detach().cpu().numpy().item(),
            'condition_action_accuracy': cond_t_accuracy.detach().cpu().numpy().item(),
            'condition_demo_accuracy': cond_p_accuracy.detach().cpu().numpy().item(),
            'decoder_greedy_token_accuracy': greedy_t_accuracy.detach().cpu().numpy().item(),
            'decoder_greedy_program_accuracy': greedy_p_accuracy.detach().cpu().numpy().item(),
            'condition_greedy_action_accuracy': greedy_a_accuracy.detach().cpu().numpy().item(),
            'condition_greedy_demo_accuracy': greedy_d_accuracy.detach().cpu().numpy().item(),
            'total_loss': loss.detach().cpu().numpy().item(),
            'rec_loss': rec_loss.detach().cpu().numpy().item(),
            'lat_loss': lat_loss.detach().cpu().numpy().item(),
            'condition_loss': condition_loss.detach().cpu().numpy().item(),
            'gt_programs': programs.detach().cpu().numpy(),
            'pred_programs': pred_programs.detach().cpu().numpy(),
            'generated_programs': generated_programs,
            'program_ids': ids,
            'latent_vectors': z.detach().cpu().numpy().tolist()
        }

    def _run_epoch(self, data_loader: DataLoader, training: bool, epoch: int) -> dict:
        epoch_info = {}
        num_batches = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            batch_info = self._run_batch(batch, training)
            logging.info(f"epoch {epoch}, batch {batch_idx}/{num_batches} loss = {batch_info['total_loss']}")
            if not training:
                for i in range(min(batch_info['gt_programs'].shape[0], 5)):
                    gt_prog = Parser.list_to_tokens(batch_info['gt_programs'][i])
                    pred_prog = Parser.list_to_tokens(batch_info['pred_programs'][i])
                    logging.info(f"gt_prog: {gt_prog}")
                    logging.info(f"pred_prog: {pred_prog}")
        # TODO: include mean of each batch_info in epoch_info
        return epoch_info

    def train(self, train_dataloader, val_dataloader, max_epoch) -> None:

        # TODO: include SupervisedRLModel.train

        best_valid_loss = np.inf
        best_valid_epoch = 0

        for epoch in range(max_epoch):
            train_info = self._run_epoch(train_dataloader, True, epoch)
            if val_dataloader is not None:
                val_info = self._run_epoch(val_dataloader, False, epoch)