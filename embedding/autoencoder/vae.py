import torch
import torch.nn as nn
import numpy as np
from embedding.autoencoder.decoder import Decoder
from embedding.autoencoder.encoder import Encoder


class VAE(nn.Module):

    def __init__(self, num_inputs, num_program_tokens, **kwargs):
        super(VAE, self).__init__()
        self._two_head = kwargs['two_head']
        self._vanilla_ae = kwargs['AE']
        self._tanh_after_mu_sigma = kwargs['net']['tanh_after_mu_sigma']
        self._tanh_after_sample = kwargs['net']['tanh_after_sample']
        self._use_latent_dist = not kwargs['net']['controller']['use_decoder_dist']
        self._rnn_type = kwargs['net']['rnn_type']
        num_outputs = num_inputs

        self.encoder = Encoder(num_inputs, num_outputs, recurrent=kwargs['recurrent_policy'],
                               hidden_size=kwargs['num_lstm_cell_units'], rnn_type=kwargs['net']['rnn_type'],
                               two_head=kwargs['two_head'])
        self.decoder = Decoder(num_inputs, num_outputs, recurrent=kwargs['recurrent_policy'],
                               hidden_size=kwargs['num_lstm_cell_units'], rnn_type=kwargs['net']['rnn_type'],
                               num_program_tokens=num_program_tokens, **kwargs)
        self._enc_mu = torch.nn.Linear(kwargs['num_lstm_cell_units'], kwargs['num_lstm_cell_units'])
        self._enc_log_sigma = torch.nn.Linear(kwargs['num_lstm_cell_units'], kwargs['num_lstm_cell_units'])
        self.tanh = torch.nn.Tanh()

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
        if self._tanh_after_mu_sigma: #False by default
            mu = self.tanh(mu)
            sigma = self.tanh(sigma)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * torch.autograd.Variable(std_z, requires_grad=False)  # Reparameterization trick

    @staticmethod
    def latent_loss(z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def forward(self, programs, program_masks, teacher_enforcing, deterministic=True):
        program_lens = program_masks.squeeze().sum(dim=-1)
        _, h_enc = self.encoder(programs, program_lens)

        if self._rnn_type == 'GRU':
            z = h_enc.squeeze() if self._vanilla_ae else self._sample_latent(h_enc.squeeze())
        elif self._rnn_type == 'LSTM':
            z = h_enc[0].squeeze() if self._vanilla_ae else self._sample_latent(h_enc[0].squeeze())
        else:
            raise NotImplementedError()

        if self._tanh_after_sample:
            z = self.tanh(z)
        return self.decoder(programs, z, teacher_enforcing=teacher_enforcing, deterministic=deterministic), z