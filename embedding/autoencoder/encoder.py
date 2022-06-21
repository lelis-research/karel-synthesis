import torch.nn as nn
from embedding.autoencoder.nn_base import NNBase


class Encoder(NNBase):
    def __init__(self, num_inputs, num_outputs, recurrent=True, hidden_size=64, rnn_type='GRU', two_head=False):
        super(Encoder, self).__init__(recurrent, num_inputs, hidden_size, rnn_type)

        self._rnn_type = rnn_type
        self._two_head = two_head
        self.token_encoder = nn.Embedding(num_inputs, num_inputs)

    def forward(self, src, src_len):
        program_embeddings = self.token_encoder(src)
        src_len = src_len.cpu()
        packed_embedded = nn.utils.rnn.pack_padded_sequence(program_embeddings, src_len, batch_first=True,
                                                            enforce_sorted=False)

        if self.is_recurrent:
            x, rnn_hxs = self.gru(packed_embedded)

        return x, rnn_hxs