import copy
import torch
import torch.nn as nn
import numpy as np
from dsl.syntax_checker import PySyntaxChecker
from embedding.autoencoder.nn_base import NNBase
from embedding.distributions import FixedCategorical
from embedding.utils import init, masked_mean, masked_sum, unmask_idx


class Decoder(NNBase):
    def __init__(self, num_inputs, num_outputs, recurrent=False, hidden_size=64, rnn_type='GRU', two_head=False, **kwargs):
        super(Decoder, self).__init__(recurrent, num_inputs+hidden_size, hidden_size, rnn_type)

        self._rnn_type = rnn_type
        self._two_head = two_head
        self.num_inputs = num_inputs
        self.max_program_len = kwargs['dsl']['max_program_len']
        self.grammar = kwargs['grammar']
        self.num_program_tokens = kwargs['num_program_tokens']
        self.setup = kwargs['algorithm']
        self.rl_algorithm = kwargs['rl']['algo']['name']
        self.value_method = kwargs['rl']['value_method']
        self.value_embedding = 'eop_rnn'

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.token_encoder = nn.Embedding(num_inputs, num_inputs)

        self.token_output_layer = nn.Sequential(
            init_(nn.Linear(hidden_size + num_inputs + hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, num_outputs)))

        # TODO: this comment is originally from LEAPS repo: check if it is actually necessary
        # This check is required only to support backward compatibility to pre-trained models
        if (self.setup =='RL' or self.setup =='supervisedRL') and kwargs['rl']['algo']['name'] != 'reinforce':
            self.critic = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic_linear = init_(nn.Linear(hidden_size, 1))

        if self._two_head:
            self.eop_output_layer = nn.Sequential(
                init_(nn.Linear(hidden_size + num_inputs + hidden_size, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, 2)))

        self._init_syntax_checker(kwargs)

        self.softmax = nn.LogSoftmax(dim=-1)

        self.train()

    def _init_syntax_checker(self, config):
        # use syntax checker to check grammar of output program prefix
        syntax_checker_tokens = copy.copy(config['dsl_tokens'])

        syntax_checker_tokens.append('<pad>')
        T2I = {token: i for i, token in enumerate(syntax_checker_tokens)}
        # T2I['<pad>'] = len(syntax_checker_tokens)
        self.T2I = T2I
        self.syntax_checker = PySyntaxChecker(T2I, use_cuda='cuda' in config['device'])

    def _forward_one_pass(self, current_tokens, context, rnn_hxs, masks):
        token_embedding = self.token_encoder(current_tokens)
        inputs = torch.cat((token_embedding, context), dim=-1)

        if self.is_recurrent:
            outputs, rnn_hxs = self._forward_rnn(inputs, rnn_hxs, masks.view(-1, 1))

        pre_output = torch.cat([outputs, token_embedding, context], dim=1)
        output_logits = self.token_output_layer(pre_output)

        value = None
        if (self.setup =='RL' or self.setup =='supervisedRL') and self.rl_algorithm != 'reinforce':
            hidden_critic = self.critic(rnn_hxs)
            value = self.critic_linear(hidden_critic)

        eop_output_logits = None
        if self._two_head:
            eop_output_logits = self.eop_output_layer(pre_output)
        return value, output_logits, rnn_hxs, eop_output_logits

    def _temp_init(self, batch_size, device):
        # create input with token as DEF
        inputs = torch.ones((batch_size)).to(torch.long).to(device)
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

        # If m) is not part of next valid tokens in syntax_mask then only eop action can be eop=0 otherwise not
        # use absence of m) to mask out eop = 1, use presence of m) and eop=1 to mask out all tokens except m)
        eop_syntax_mask = None
        if self._two_head:
            # use absence of m) to mask out eop = 1
            gather_m_closed = torch.tensor(batch_size * [self.T2I['m)']], dtype=torch.long, device=device).view(-1, 1)
            eop_in_valid_set = torch.gather(syntax_mask, 1, gather_m_closed)
            eop_syntax_mask = torch.zeros((batch_size, 2), device=device)
            # if m) is absent we can't predict eop=1
            eop_syntax_mask[:, 1] = eop_in_valid_set.flatten()

        return syntax_mask, eop_syntax_mask, grammar_state

    def _get_eop_preds(self, eop_output_logits, eop_syntax_mask, syntax_mask, output_mask, deterministic=False):
        batch_size = eop_output_logits.shape[0]
        device = eop_output_logits.device

        # eop_action
        if eop_syntax_mask is not None:
            assert eop_output_logits.shape == eop_syntax_mask.shape
            eop_output_logits += eop_syntax_mask
        if self.setup == 'supervised':
            eop_preds = self.softmax(eop_output_logits).argmax(dim=-1).to(torch.bool)
        elif self.setup == 'RL':
            # define distribution over current logits
            eop_dist = FixedCategorical(logits=eop_output_logits)
            # sample actions
            eop_preds = eop_dist.mode() if deterministic else eop_dist.sample()
        else:
            raise NotImplementedError()


        #  use presence of m) and eop=1 to mask out all tokens except m)
        new_output_mask = (~(eop_preds.to(torch.bool))) * output_mask
        assert output_mask.dtype == torch.bool
        output_mask_change = (new_output_mask != output_mask).view(-1, 1)
        output_mask_change_repeat = output_mask_change.repeat(1, syntax_mask.shape[1])
        new_syntax_mask = -torch.finfo(torch.float32).max * torch.ones_like(syntax_mask).float()
        new_syntax_mask[:, self.T2I['m)']] = 0
        syntax_mask = torch.where(output_mask_change_repeat, new_syntax_mask, syntax_mask)

        return eop_preds, eop_output_logits, syntax_mask

    def forward(self, gt_programs, embeddings, teacher_enforcing=True, action=None, output_mask_all=None,
                eop_action=None, deterministic=False, evaluate=False, max_program_len=float('inf')):
        if self.setup == 'supervised':
            assert deterministic == True
        batch_size, device = embeddings.shape[0], embeddings.device
        # NOTE: for pythorch >=1.2.0, ~ only works correctly on torch.bool
        if evaluate:
            output_mask = output_mask_all[:, 0]
        else:
            output_mask = torch.ones(batch_size).to(torch.bool).to(device)

        current_tokens, gru_mask = self._temp_init(batch_size, device)
        if self._rnn_type == 'GRU':
            rnn_hxs = embeddings
        elif self._rnn_type == 'LSTM':
            rnn_hxs = (embeddings, embeddings)
        else:
            raise NotImplementedError()

        # Encode programs
        max_program_len = min(max_program_len, self.max_program_len)
        value_all = []
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
            value, output_logits, rnn_hxs, eop_output_logits = self._forward_one_pass(current_tokens, embeddings,
                                                                                      rnn_hxs, gru_mask)

            # limit possible actions using syntax checker if available
            # action_logits * syntax_mask where syntax_mask = {-inf, 0}^|num_program_tokens|
            # syntax_mask = 0  for action a iff for given input(e.g.'DEF'), a(e.g.'run') creates a valid program prefix
            syntax_mask = None
            eop_syntax_mask = None
            mask_size = output_logits.shape[1]
            syntax_mask, eop_syntax_mask, grammar_state = self._get_syntax_mask(batch_size, current_tokens,
                                                                                mask_size, grammar_state)

            # get eop action and new syntax mask if using syntax checker
            if self._two_head:
                eop_preds, eop_output_logits, syntax_mask = self._get_eop_preds(eop_output_logits, eop_syntax_mask,
                                                                                syntax_mask, output_mask_all[:, i])

            # apply softmax
            if syntax_mask is not None:
                assert (output_logits.shape == syntax_mask.shape) or self.setup == 'CEM', '{}:{}'.format(output_logits.shape, syntax_mask.shape)
                output_logits += syntax_mask
            if self.setup == 'supervised' or self.setup == 'CEM':
                preds = self.softmax(output_logits).argmax(dim=-1)
            elif self.setup == 'RL':
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

                if self._two_head:
                    raise NotImplementedError()
                # calculate entropy
                dist_entropy = dist.entropy()
                if self._two_head:
                    raise NotImplementedError()
                pred_programs_log_probs_all.append(pred_programs_log_probs)
                dist_entropy_all.append(dist_entropy.view(-1, 1))
            else:
                raise NotImplementedError()

            # calculate mask for current tokens
            assert preds.shape == output_mask.shape
            if not evaluate:
                if self._two_head:
                    output_mask = (~(eop_preds.to(torch.bool))) * output_mask
                else:
                    output_mask = (~((preds == self.num_program_tokens - 1).to(torch.bool))) * output_mask

                # recalculate first occurrence of <pad> for each program
                first_end_token_idx = torch.min(first_end_token_idx,
                                                ((self.max_program_len * output_mask.float()) +
                                                 ((1 - output_mask.float()) * i)).flatten())

            value_all.append(value)
            output_logits_all.append(output_logits)
            pred_programs.append(preds)
            if self._two_head:
                eop_output_logits_all.append(eop_output_logits)
                eop_pred_programs.append(eop_preds)
            if not evaluate:
                output_mask_all[:, i] = output_mask.flatten()

            if self.setup == 'supervised':
                if teacher_enforcing:
                    current_tokens = gt_programs[:, i+1].squeeze()
                else:
                    current_tokens = preds.squeeze()
            else:
                if evaluate:
                    assert self.setup == 'RL'
                    current_tokens = action[:, i]
                else:
                    current_tokens = preds.squeeze()


        # umask first end-token for two headed policy
        if not evaluate:
            output_mask_all = unmask_idx(output_mask_all, first_end_token_idx, self.max_program_len).detach()

        # combine all token parameters to get program parameters
        raw_pred_programs_all = torch.stack(pred_programs, dim=1)
        raw_output_logits_all = torch.stack(output_logits_all, dim=1)
        pred_programs_len = torch.sum(output_mask_all, dim=1, keepdim=True)

        if not self._two_head:
            assert output_mask_all.dtype == torch.bool
            pred_programs_all = torch.where(output_mask_all, raw_pred_programs_all,
                                            int(self.num_program_tokens - 1) * torch.ones_like(raw_pred_programs_all))
            eop_pred_programs_all = -1 * torch.ones_like(pred_programs_all)
            raw_eop_output_logits_all = None
        else:
            pred_programs_all = raw_pred_programs_all
            eop_pred_programs_all = torch.stack(eop_pred_programs, dim=1)
            raw_eop_output_logits_all = torch.stack(eop_output_logits_all, dim=1)

        # calculate log_probs, value, actions for program from token values
        if self.setup == 'RL':
            raw_pred_programs_log_probs_all = torch.cat(pred_programs_log_probs_all, dim=1)
            pred_programs_log_probs_all = masked_sum(raw_pred_programs_log_probs_all, output_mask_all,
                                                     dim=1, keepdim=True)

            raw_dist_entropy_all = torch.cat(dist_entropy_all, dim=1)
            dist_entropy_all = masked_mean(raw_dist_entropy_all, output_mask_all,
                                           dim=tuple(range(len(output_mask_all.shape))))

            # calculate value for program from token values
            if self.rl_algorithm != 'reinforce':
                if self.value_method == 'mean':
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_all = masked_mean(raw_value_all, output_mask_all, dim=1, keepdim=True)
                else:
                    # calculate value function from hidden states
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_idx = torch.sum(output_mask_all, dim=1, keepdim=True) - 1
                    assert len(value_idx.shape) == 2 and value_idx.shape[1] == 1
                    value_all = torch.gather(raw_value_all, 1, value_idx)

                    # This value calculation is just for sanity check
                    with torch.no_grad():
                        value_idx_2 = first_end_token_idx.clamp(max=self.max_program_len - 1).long().reshape(-1, 1)
                        value_all_2 = torch.gather(raw_value_all, 1, value_idx_2)
                        assert torch.sum(value_all != value_all_2) == 0
                    assert value_all.shape[0] == batch_size
            else:
                value_all = torch.zeros_like(pred_programs_log_probs_all)
        else:
            dist_entropy_all = None
            value_all = None

        return value_all, pred_programs_all, pred_programs_len, pred_programs_log_probs_all, raw_output_logits_all,\
               eop_pred_programs_all, raw_eop_output_logits_all, output_mask_all, dist_entropy_all