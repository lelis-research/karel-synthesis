import torch
import torch.nn as nn
import numpy as np
from dsl.production import Production
from embedding.autoencoder.decoder import Decoder
from embedding.autoencoder.encoder import Encoder
from embedding.config.config import Config


class VAE(nn.Module):

    def __init__(self, num_inputs, dsl: Production, device: torch.device, config: Config):
        super(VAE, self).__init__()
        num_outputs = num_inputs

        self.encoder = Encoder(num_inputs, num_outputs, config)
        self.decoder = Decoder(num_inputs, num_outputs, dsl, device, config)
        self._enc_mu = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self._enc_log_sigma = torch.nn.Linear(config.hidden_size, config.hidden_size)

    @property
    def latent_dim(self):
        return  self._enc_mu.out_features

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).to(torch.float).to(h_enc.device)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * torch.autograd.Variable(std_z, requires_grad=False)  # Reparameterization trick

    @staticmethod
    def latent_loss(z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def forward(self, programs, program_masks, teacher_enforcing, deterministic=True, reinforce_step=False):
        program_lens = program_masks.squeeze().sum(dim=-1)
        _, h_enc = self.encoder(programs, program_lens)
        z = self._sample_latent(h_enc.squeeze())
        return self.decoder(
            programs, z, teacher_enforcing=teacher_enforcing,
            deterministic=deterministic, reinforce_step=reinforce_step
        ), z