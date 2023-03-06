import numpy as np
import torch
from torch import nn
from dsl.production import Production
from dsl.syntax_checker import PySyntaxChecker
from karel.world import STATE_TABLE
from karel.world_batch import WorldBatch
from embedding.config.config import Config
from embedding.utils import init
from typing import NamedTuple


class ModelOutput(NamedTuple):
    pred_progs: torch.Tensor
    pred_progs_logits: torch.Tensor
    pred_progs_masks: torch.Tensor
    pred_a_h: torch.Tensor
    pred_a_h_logits: torch.Tensor
    pred_a_h_masks: torch.Tensor


class BaseVAE(nn.Module):
    def __init__(self, dsl: Production, device: torch.device, config: Config) -> None:
        super().__init__()
        
        self.device = device
        
        self.max_demo_length = config.max_demo_length
        self.max_program_length = config.max_program_len
        
        # Z
        self.hidden_size = config.hidden_size
        
        # A
        self.num_agent_actions = len(dsl.get_actions()) + 1 # +1 because we have a NOP action
        
        # T
        self.num_program_tokens = len(dsl.get_tokens()) + 1 # +1 because we have a <pad> token
        
        self.init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                    constant_(x, 0), nn.init.calculate_gain('relu'))
        
        # CxHxW
        self.state_shape = (len(STATE_TABLE), config.env_height, config.env_width)
        
        # Input: s_i (CxHxW). Output: enc(s_i) (Z).
        self.state_encoder = nn.Sequential(
            self.init_(nn.Conv2d(self.state_shape[0], 32, 3, stride=1)), nn.ReLU(),
            self.init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(), nn.Flatten(),
            self.init_(nn.Linear(32 * 4 * 4, self.hidden_size)), nn.ReLU()
        )
        
        # Input: a_i (A). Output: enc(a_i) (A).
        self.action_encoder = nn.Embedding(self.num_agent_actions, self.num_agent_actions)
        
        # Input: rho_i (T). Output: enc(rho_i) (T).
        self.token_encoder = nn.Embedding(self.num_program_tokens, self.num_program_tokens)
        
        # Encoder VAE utils
        self.encoder_mu = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.encoder_log_sigma = torch.nn.Linear(config.hidden_size, config.hidden_size)
        
        self.softmax = nn.LogSoftmax(dim=-1)
        
        syntax_checker_tokens = dsl.get_tokens()
        syntax_checker_tokens.append('<pad>')
        self.T2I = {token: i for i, token in enumerate(syntax_checker_tokens)}
        self.syntax_checker = PySyntaxChecker(self.T2I, self.device)

    def env_init(self, states: torch.Tensor):
        states = states.detach().cpu().numpy().astype(np.bool_)
        # C x H x W to H x W x C
        states = np.moveaxis(states,[-1,-2,-3], [-2,-3,-1])
        self._world = WorldBatch(states)

    def env_step(self, states: torch.Tensor, actions: torch.Tensor):
        states = states.detach().cpu().numpy().astype(np.bool_)
        # C x H x W to H x W x C
        states = np.moveaxis(states,[-1,-2,-3], [-2,-3,-1])
        assert states.shape[-1] == 16
        # karel world expects H x W x C
        new_states = self._world.step(actions.detach().cpu().numpy())
        new_states = np.moveaxis(new_states,[-1,-2,-3], [-3,-1,-2])
        new_states = torch.tensor(new_states, dtype=torch.float32, device=self.device)
        return new_states

    def sample_latent_vector(self, enc_hidden_state: torch.Tensor) -> torch.Tensor:
        # Sampling z with reperameterization trick
        mu = self.encoder_mu(enc_hidden_state)
        log_sigma = self.encoder_log_sigma(enc_hidden_state)
        sigma = torch.exp(log_sigma)
        std_z = torch.randn(sigma.size(), device=self.device)
        
        z = mu + sigma * torch.autograd.Variable(std_z, requires_grad=False)
        
        self.z_mu = mu
        self.z_sigma = sigma
        
        return z
    
    def get_latent_loss(self):
        mean_sq = self.z_mu * self.z_mu
        stddev_sq = self.z_sigma * self.z_sigma
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
    
    def get_syntax_mask(self, batch_size: int, current_tokens: torch.Tensor, grammar_state: list):
        out_of_syntax_list = []
        out_of_syntax_mask = torch.zeros((batch_size, self.num_program_tokens), dtype=torch.bool, device=self.device)

        for program_idx, inp_token in enumerate(current_tokens):
            inp_dsl_token = inp_token.detach().cpu().numpy().item()
            out_of_syntax_list.append(self.syntax_checker.get_sequence_mask(grammar_state[program_idx],
                                                                            [inp_dsl_token]).to(self.device))
        torch.cat(out_of_syntax_list, 0, out=out_of_syntax_mask)
        out_of_syntax_mask = out_of_syntax_mask.squeeze()
        syntax_mask = torch.where(out_of_syntax_mask,
                                  -torch.finfo(torch.float32).max * torch.ones_like(out_of_syntax_mask).float(),
                                  torch.zeros_like(out_of_syntax_mask).float())

        return syntax_mask, grammar_state

    def forward(self, s_h: torch.Tensor, a_h: torch.Tensor, a_h_mask: torch.Tensor, 
                prog: torch.Tensor, prog_mask: torch.Tensor, prog_teacher_enforcing = True,
                a_h_teacher_enforcing = True) -> ModelOutput:
        raise NotImplementedError