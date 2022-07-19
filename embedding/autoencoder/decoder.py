import copy
import torch
import torch.nn as nn
import numpy as np
from dsl.production import Production
from dsl.syntax_checker import PySyntaxChecker
from embedding.config.config import Config
from embedding.distributions import FixedCategorical
from embedding.utils import init, masked_mean, masked_sum, unmask_idx


class Decoder(nn.Module):
    def __init__(self, num_inputs, num_outputs, dsl: Production, config: Config):
        # super(Decoder, self).__init__(recurrent, num_inputs+hidden_size, hidden_size, rnn_type)
        super(Decoder, self).__init__()

        self.gru = nn.GRU(num_inputs+config.hidden_size, config.hidden_size)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.num_inputs = num_inputs
        self.max_program_len = config.max_program_len
        self.num_program_tokens = len(dsl.get_tokens()) + 1

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self.token_encoder = nn.Embedding(num_inputs, num_inputs)

        self.token_output_layer = nn.Sequential(
            init_(nn.Linear(config.hidden_size + num_inputs + config.hidden_size, config.hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(config.hidden_size, num_outputs))
        )

        self._init_syntax_checker(dsl, config.device)

        self.softmax = nn.LogSoftmax(dim=-1)

        self.train() # TODO: is this needed?

    def _init_syntax_checker(self, dsl: Production, device: str):
        # use syntax checker to check grammar of output program prefix
        # syntax_checker_tokens = copy.copy(dsl.get_tokens())
        syntax_checker_tokens = dsl.get_tokens()
        syntax_checker_tokens.append('<pad>')
        self.T2I = {token: i for i, token in enumerate(syntax_checker_tokens)}
        self.syntax_checker = PySyntaxChecker(self.T2I, use_cuda='cuda' in device)

    def _forward_one_pass(self, current_tokens, context, rnn_hxs, masks):
        token_embedding = self.token_encoder(current_tokens)
        inputs = torch.cat((token_embedding, context), dim=-1)

        # TODO: do I need to unsqueeze/squeeze?
        outputs, rnn_hxs = self.gru(inputs.unsqueeze(0), (rnn_hxs * masks.view(-1, 1)).unsqueeze(0))
        outputs = outputs.squeeze(0)
        rnn_hxs = rnn_hxs.squeeze(0)

        pre_output = torch.cat([outputs, token_embedding, context], dim=1)
        output_logits = self.token_output_layer(pre_output)

        return output_logits, rnn_hxs

    def _temp_init(self, batch_size, device):
        # create input with token as DEF
        inputs = torch.ones((batch_size)).to(torch.long).to(device)
        # TODO: I don't understand this line: is it needed?
        inputs = (0 * inputs)# if self.use_simplified_dsl else (2 * inputs)

        # input to the GRU
        gru_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        return inputs, gru_mask

    def _get_syntax_mask(self, batch_size, current_tokens, mask_size, grammar_state):
        out_of_syntax_list = []
        device = current_tokens.device
        out_of_syntax_mask = torch.zeros((batch_size, mask_size),
                                         dtype=torch.bool, device=device)

        for program_idx, inp_token in enumerate(current_tokens):
            inp_dsl_token = inp_token.detach().cpu().numpy().item()
            out_of_syntax_list.append(self.syntax_checker.get_sequence_mask(grammar_state[program_idx],
                                                                            [inp_dsl_token]).to(device))
        torch.cat(out_of_syntax_list, 0, out=out_of_syntax_mask)
        out_of_syntax_mask = out_of_syntax_mask.squeeze()
        syntax_mask = torch.where(out_of_syntax_mask,
                                  -torch.finfo(torch.float32).max * torch.ones_like(out_of_syntax_mask).float(),
                                  torch.zeros_like(out_of_syntax_mask).float())

        return syntax_mask, grammar_state

    def forward(self, gt_programs, embeddings, teacher_enforcing=True, action=None, output_mask_all=None,
                deterministic=False, evaluate=False, max_program_len=float('inf'), reinforce_step=False):
        
        batch_size, device = embeddings.shape[0], embeddings.device
        # NOTE: for pythorch >=1.2.0, ~ only works correctly on torch.bool
        if evaluate:
            output_mask = output_mask_all[:, 0]
        else:
            output_mask = torch.ones(batch_size).to(torch.bool).to(device)

        current_tokens, gru_mask = self._temp_init(batch_size, device)
        rnn_hxs = embeddings

        # Encode programs
        max_program_len = min(max_program_len, self.max_program_len)
        pred_programs = []
        pred_programs_log_probs_all = []
        dist_entropy_all = []
        eop_dist_entropy_all = []
        output_logits_all = []
        eop_output_logits_all = []
        eop_pred_programs = []
        if not evaluate:
            output_mask_all = torch.ones(batch_size, self.max_program_len).to(torch.bool).to(device)
        first_end_token_idx = self.max_program_len * torch.ones(batch_size).to(device)

        # using get_initial_checker_state2 because we skip prediction for 'DEF', 'run' tokens
        grammar_state = [self.syntax_checker.get_initial_checker_state()
                            for _ in range(batch_size)]

        for i in range(max_program_len):
            output_logits, rnn_hxs = self._forward_one_pass(
                current_tokens, embeddings, rnn_hxs, gru_mask
            )

            # limit possible actions using syntax checker if available
            # action_logits * syntax_mask where syntax_mask = {-inf, 0}^|num_program_tokens|
            # syntax_mask = 0  for action a iff for given input(e.g.'DEF'), a(e.g.'run') creates a valid program prefix
            syntax_mask = None
            mask_size = output_logits.shape[1]
            syntax_mask, grammar_state = self._get_syntax_mask(batch_size, current_tokens,
                                                                                mask_size, grammar_state)

            # apply softmax
            if syntax_mask is not None:
                output_logits += syntax_mask

            if reinforce_step:
                # define distribution over current logits
                dist = FixedCategorical(logits=output_logits)
                # sample actions
                preds = dist.mode().squeeze() if deterministic else dist.sample().squeeze()
                # calculate log probabilities
                if evaluate:
                    assert action[:,i].shape == preds.shape
                    pred_programs_log_probs = dist.log_probs(action[:,i])
                else:
                    pred_programs_log_probs = dist.log_probs(preds)

                # calculate entropy
                dist_entropy = dist.entropy()
                pred_programs_log_probs_all.append(pred_programs_log_probs)
                dist_entropy_all.append(dist_entropy.view(-1, 1))
            else:
                preds = self.softmax(output_logits).argmax(dim=-1)

            # calculate mask for current tokens
            assert preds.shape == output_mask.shape
            if not evaluate:
                output_mask = (~((preds == self.num_program_tokens - 1).to(torch.bool))) * output_mask

                # recalculate first occurrence of <pad> for each program
                first_end_token_idx = torch.min(first_end_token_idx,
                                                ((self.max_program_len * output_mask.float()) +
                                                 ((1 - output_mask.float()) * i)).flatten())

            # value_all.append(value)
            output_logits_all.append(output_logits)
            pred_programs.append(preds)
            if not evaluate:
                output_mask_all[:, i] = output_mask.flatten()

            if reinforce_step:
                if evaluate:
                    current_tokens = action[:, i]
                else:
                    current_tokens = preds.squeeze()
            else:
                if teacher_enforcing:
                    current_tokens = gt_programs[:, i+1].squeeze()
                else:
                    current_tokens = preds.squeeze()

        # TODO: do we need this block? (prob not)
        # umask first end-token for two headed policy
        if not evaluate:
            output_mask_all = unmask_idx(output_mask_all, first_end_token_idx, self.max_program_len).detach()

        # combine all token parameters to get program parameters
        raw_pred_programs_all = torch.stack(pred_programs, dim=1)
        raw_output_logits_all = torch.stack(output_logits_all, dim=1)
        pred_programs_len = torch.sum(output_mask_all, dim=1, keepdim=True)

        assert output_mask_all.dtype == torch.bool
        pred_programs_all = torch.where(output_mask_all, raw_pred_programs_all,
                                        int(self.num_program_tokens - 1) * torch.ones_like(raw_pred_programs_all))
        eop_pred_programs_all = -1 * torch.ones_like(pred_programs_all)
        raw_eop_output_logits_all = None

        # calculate log_probs, value, actions for program from token values
        if reinforce_step:
            raw_pred_programs_log_probs_all = torch.cat(pred_programs_log_probs_all, dim=1)
            pred_programs_log_probs_all = masked_sum(raw_pred_programs_log_probs_all, output_mask_all,
                                                     dim=1, keepdim=True)

            raw_dist_entropy_all = torch.cat(dist_entropy_all, dim=1)
            dist_entropy_all = masked_mean(raw_dist_entropy_all, output_mask_all,
                                           dim=tuple(range(len(output_mask_all.shape))))

        else:
            dist_entropy_all = None

        return pred_programs_all, pred_programs_len, pred_programs_log_probs_all, raw_output_logits_all,\
               eop_pred_programs_all, raw_eop_output_logits_all, output_mask_all, dist_entropy_all