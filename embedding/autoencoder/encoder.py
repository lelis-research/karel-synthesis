import torch
import torch.nn as nn
from embedding.config.config import Config
from embedding.utils import init


class Encoder(nn.Module):
    
    def __init__(self, num_inputs: int, num_outputs: int, config: Config):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(num_inputs, config.hidden_size)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.token_encoder = nn.Embedding(num_inputs, num_inputs)

    def forward(self, src: torch.Tensor, src_len: torch.Tensor):
        program_embeddings = self.token_encoder(src)
        src_len = src_len.cpu()
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            program_embeddings, src_len, batch_first=True, enforce_sorted=False
        )
        x, rnn_hxs = self.gru(packed_embedded)
        return x, rnn_hxs