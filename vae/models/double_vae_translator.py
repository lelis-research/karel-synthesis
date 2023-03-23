import torch
from torch import nn

from config import Config
from dsl import DSL
from .double_vae import DoubleVAE

class DoubleVAETranslator(DoubleVAE):
    
    def __init__(self, dsl: DSL, device: torch.device,
                 double_vae_params_file = 'output/double_vae_128/model/best_val.ptp') -> None:
        super().__init__(dsl, device)
        
        params = torch.load(double_vae_params_file, map_location=device)
        self.load_state_dict(params, strict=False)
        
        # z_a_h -> z_prog
        self.translator_mlp = nn.Sequential(
            self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.Tanh(),
            self.init_(nn.Linear(self.hidden_size, self.hidden_size))            
        )
        
        self.transl_loss_fn = nn.MSELoss(reduction='mean')
        
        self.to(self.device)
        
    def translate_latent(self, z_a_h: torch.Tensor) -> torch.Tensor:
        return self.translator_mlp(z_a_h)
    
    def forward(self, s_h: torch.Tensor, a_h: torch.Tensor, a_h_mask: torch.Tensor, prog: torch.Tensor,
                prog_mask: torch.Tensor, prog_teacher_enforcing=True, a_h_teacher_enforcing=True):
        z_prog = self.encode_prog(prog, prog_mask).detach()
        
        z_a_h = self.encode_a_h(s_h, a_h, a_h_mask).detach()
        
        z_prog_pred = self.translate_latent(z_a_h)
        
        return z_prog_pred, z_prog
        
        