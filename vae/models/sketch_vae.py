from typing import Union
import torch
import torch.nn as nn

from dsl import DSL

from ..utils import init_gru
from .base_vae import BaseVAE, ModelOutput


class SketchVAE(BaseVAE):
    
    def __init__(self, dsl: DSL, device: torch.device, hidden_size: Union[int, None] = None):
        super().__init__(dsl.extend_dsl(), device, hidden_size)
        
        # Inputs: enc(rho_i) (T). Output: enc_state (Z). Hidden state: h_i: z = h_t (Z).
        self.encoder_gru = nn.GRU(self.num_program_tokens, self.hidden_size)
        init_gru(self.encoder_gru)
        
        # Inputs: enc(rho_i) (T), z (Z). Output: dec_state (Z). Hidden state: h_i: h_0 = z (Z).
        self.decoder_gru = nn.GRU(self.hidden_size + self.num_program_tokens, self.hidden_size)
        init_gru(self.decoder_gru)
        
        # Input: dec_state (Z), z (Z), enc(rho_i) (T). Output: prob(rho_hat) (T).
        self.decoder_mlp = nn.Sequential(
            self.init_(nn.Linear(2 * self.hidden_size + self.num_program_tokens, self.hidden_size)),
            nn.Tanh(), self.init_(nn.Linear(self.hidden_size, self.num_program_tokens))
        )
        
        self.to(self.device)
        
    def encode(self, progs: torch.Tensor, progs_mask: torch.Tensor, prog_teacher_enforcing = True):
        if len(progs.shape) == 3:
            batch_size, demos_per_program, _ = progs.shape
            progs = progs.view(batch_size * demos_per_program, -1)
            progs_mask = progs_mask.view(batch_size * demos_per_program, -1)
        
        progs_len = progs_mask.squeeze(-1).sum(dim=-1).cpu()
        
        enc_progs = self.token_encoder(progs)
        
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            enc_progs, progs_len, batch_first=True, enforce_sorted=False
        )
        
        _, enc_hidden_state = self.encoder_gru(packed_inputs)
        enc_hidden_state = enc_hidden_state.squeeze(0)
        
        z = self.sample_latent_vector(enc_hidden_state)
        
        return z
    
    def decode(self, z: torch.Tensor, progs: torch.Tensor, progs_mask: torch.Tensor,
               prog_teacher_enforcing = True):
        if progs is not None:
            if len(progs.shape) == 3:
                b, demos_per_program, _ = progs.shape
                progs = progs.view(b * demos_per_program, -1)
                progs_mask = progs_mask.view(b * demos_per_program, -1)
        
        batch_size, _ = z.shape
        
        gru_hidden_state = z.unsqueeze(0)
        
        # Initialize tokens as DEFs
        current_tokens = torch.zeros((batch_size), dtype=torch.long, device=self.device)
        
        grammar_state = [self.syntax_checker.get_initial_checker_state()
                         for _ in range(batch_size)]
        
        pred_progs = []
        pred_progs_logits = []
        
        for i in range(1, self.max_program_length):
            token_embedding = self.token_encoder(current_tokens)
            gru_inputs = torch.cat((token_embedding, z), dim=-1)
            gru_inputs = gru_inputs.unsqueeze(0)
            
            gru_output, gru_hidden_state = self.decoder_gru(gru_inputs, gru_hidden_state)
            
            mlp_input = torch.cat([gru_output.squeeze(0), token_embedding, z], dim=1)
            pred_token_logits = self.decoder_mlp(mlp_input)
            
            syntax_mask, grammar_state = self.get_syntax_mask(batch_size, current_tokens, grammar_state)
            
            pred_token_logits += syntax_mask
            
            pred_tokens = self.softmax(pred_token_logits).argmax(dim=-1)
            
            pred_progs.append(pred_tokens)
            pred_progs_logits.append(pred_token_logits)
            
            if prog_teacher_enforcing:
                # Enforce next token with ground truth
                current_tokens = progs[:, i].view(batch_size)
            else:
                # Pass current prediction to next iteration
                current_tokens = pred_tokens.view(batch_size)
        
        pred_progs = torch.stack(pred_progs, dim=1)
        pred_progs_logits = torch.stack(pred_progs_logits, dim=1)
        pred_progs_masks = (pred_progs != self.pad_token)
        
        return pred_progs, pred_progs_logits, pred_progs_masks
    
    def forward(self, s_h: torch.Tensor, a_h: torch.Tensor, a_h_mask: torch.Tensor, 
                prog: torch.Tensor, prog_mask: torch.Tensor, prog_teacher_enforcing = True,
                a_h_teacher_enforcing = True) -> ModelOutput:
        z = self.encode(prog, prog_mask, prog_teacher_enforcing)
        
        decoder_result = self.decode(z, prog, prog_mask, prog_teacher_enforcing)
        pred_progs, pred_progs_logits, pred_progs_masks = decoder_result
        
        return ModelOutput(pred_progs, pred_progs_logits, pred_progs_masks,
                           None, None, None)
        
    def encode_program(self, prog: torch.Tensor):
        if prog.dim() == 1:
            prog = prog.unsqueeze(0)
        
        prog_mask = (prog != self.num_program_tokens - 1)
        
        z = self.encode(prog, prog_mask, False)
        
        return z
    
    def decode_vector(self, z: torch.Tensor):
        pred_progs, _, pred_progs_masks = self.decode(z, None, None, False)
        
        pred_progs_tokens = []
        for prog, prog_mask in zip(pred_progs, pred_progs_masks):
            pred_progs_tokens.append([0] + prog[prog_mask].cpu().numpy().tolist())
        
        return pred_progs_tokens