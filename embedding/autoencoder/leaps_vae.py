import torch
import torch.nn as nn
from dsl.production import Production
from config.config import Config
from embedding.utils import init_gru
from embedding.autoencoder.base_vae import BaseVAE, ModelOutput


class LeapsVAE(BaseVAE):
    
    def __init__(self, dsl: Production, device: torch.device, config: Config):
        super().__init__(dsl, device, config)
        
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
        
        # Input: enc(rho_i) (T), z (Z). Output: prog_out (Z).
        self.policy_gru = nn.GRU(2 * self.hidden_size + self.num_agent_actions, self.hidden_size)
        init_gru(self.policy_gru)
        
        # Inputs: prog_out (Z), z (Z), enc(rho_i). Output: prob(rho_hat) (T).
        self.policy_mlp = nn.Sequential(
            self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(self.hidden_size, self.num_agent_actions))
        )
        
        self.to(self.device)
        
    def encode(self, progs: torch.Tensor, progs_mask: torch.Tensor, prog_teacher_enforcing = True):
        progs_len = progs_mask.squeeze(-1).sum(dim=-1)
        progs_len = progs_len.cpu()
        
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
        batch_size = z.shape[0]
        
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
        pred_progs_masks = (pred_progs != self.num_program_tokens - 1)
        
        return pred_progs, pred_progs_logits, pred_progs_masks
    
    def policy_executor(self, z: torch.Tensor, s_h: torch.Tensor, a_h: torch.Tensor,
                        a_h_mask: torch.Tensor, a_h_teacher_enforcing = True):
        batch_size, demos_per_program, _, c, h, w = s_h.shape
        
        # Taking only first state and squeezing over first 2 dimensions
        current_state = s_h[:, :, 0, :, :, :].view(batch_size*demos_per_program, c, h, w)
        
        current_action = a_h[:, :, 0].squeeze().long().view(-1, 1)
        
        z_repeated = z.unsqueeze(1).repeat(1, demos_per_program, 1)
        
        gru_hidden = z_repeated.view(1, batch_size*demos_per_program, z_repeated.shape[-1])
        
        pred_a_h = []
        pred_a_h_logits = []
        
        if not a_h_teacher_enforcing:
            self.env_init(current_state)
        
        terminated_policy = torch.zeros_like(current_action, dtype=torch.bool, device=self.device)
        
        mask_valid_actions = torch.tensor((self.num_agent_actions - 1) * [-torch.finfo(torch.float32).max]
                                          + [0.], device=self.device)
        
        for i in range(1, self.max_demo_length):
            enc_state = self.state_encoder(current_state)
            enc_state = enc_state.view(batch_size, demos_per_program, self.hidden_size)
            
            action_encoder_input = current_action.view(batch_size, demos_per_program)
            enc_action = self.action_encoder(action_encoder_input)
            gru_inputs = torch.cat((z_repeated, enc_state, enc_action), dim=-1)
            gru_inputs = gru_inputs.view(batch_size * demos_per_program, -1)
            gru_inputs = gru_inputs.unsqueeze(0)
            
            gru_out, gru_hidden = self.policy_gru(gru_inputs, gru_hidden)
            gru_out = gru_out.squeeze(0)
            
            pred_action_logits = self.policy_mlp(gru_out)
            
            masked_action_logits = pred_action_logits + terminated_policy * mask_valid_actions
            
            current_action = self.softmax(masked_action_logits).argmax(dim=-1).view(-1, 1)
            
            pred_a_h.append(current_action)
            pred_a_h_logits.append(pred_action_logits)
            
            # Apply teacher enforcing while training
            if a_h_teacher_enforcing:
                current_action = a_h[:, :, i].squeeze().long().view(-1, 1)
                current_state = s_h[:, :, i, :, :, :].view(batch_size*demos_per_program, c, h, w)
            # Otherwise, step in actual environment to get next state
            else:
                current_state = self.env_step(current_state, current_action)
                
            terminated_policy = torch.logical_or(current_action == self.num_agent_actions - 1,
                                                 terminated_policy)
    
        pred_a_h = torch.stack(pred_a_h, dim=1).squeeze(-1)
        pred_a_h_logits = torch.stack(pred_a_h_logits, dim=1)
        pred_a_h_masks = (pred_a_h != self.num_agent_actions - 1)
        
        return pred_a_h, pred_a_h_logits, pred_a_h_masks
    
    def forward(self, s_h: torch.Tensor, a_h: torch.Tensor, a_h_mask: torch.Tensor, 
                prog: torch.Tensor, prog_mask: torch.Tensor, prog_teacher_enforcing = True,
                a_h_teacher_enforcing = True) -> ModelOutput:
        z = self.encode(prog, prog_mask, prog_teacher_enforcing)
        
        decoder_result = self.decode(z, prog, prog_mask, prog_teacher_enforcing)
        pred_progs, pred_progs_logits, pred_progs_masks = decoder_result
        
        policy_result = self.policy_executor(z, s_h, a_h, a_h_mask, a_h_teacher_enforcing)
        pred_a_h, pred_a_h_logits, pred_a_h_masks = policy_result
        
        return ModelOutput(pred_progs, pred_progs_logits, pred_progs_masks,
                           pred_a_h, pred_a_h_logits, pred_a_h_masks)
        
    def encode_program(self, prog: torch.Tensor) -> torch.Tensor:
        if prog.dim() == 1:
            prog = prog.unsqueeze(0)
        
        prog_mask = (prog != self.num_program_tokens - 1)
        
        z = self.encode(prog, prog_mask, False)
        
        return z
    
    def decode_vector(self, z: torch.Tensor) -> torch.Tensor:
        pred_progs, _, pred_progs_masks = self.decode(z, None, None, False)
        
        return pred_progs[pred_progs_masks]