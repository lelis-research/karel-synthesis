import torch
import torch.nn as nn
import numpy as np
from embedding.autoencoder.nn_base import NNBase
from embedding.distributions import FixedCategorical
from embedding.utils import init, masked_mean, masked_sum, Flatten, unmask_idx2
from karel.data import Data
from karel.environment import Environment
from karel.world import World


class ConditionPolicy(NNBase):
    def __init__(self, envs, **kwargs):
        hidden_size = kwargs['num_lstm_cell_units']
        rnn_type = kwargs['net']['rnn_type']
        recurrent = kwargs['recurrent_policy']
        self.num_agent_actions = kwargs['dsl']['num_agent_actions']
        super(ConditionPolicy, self).__init__(recurrent, 2 * hidden_size + self.num_agent_actions, hidden_size, rnn_type)

        self.envs = envs
        self.state_shape = (16, kwargs['height'], kwargs['width'])
        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._rnn_type = rnn_type
        self.max_demo_length = kwargs['max_demo_length']
        self.setup = kwargs['algorithm']
        self.rl_algorithm = kwargs['rl']['algo']['name']
        self.value_method = kwargs['rl']['value_method']
        self.value_embedding = 'eop_rnn'
        self.use_teacher_enforcing =  kwargs['net']['condition']['use_teacher_enforcing']
        self.states_source = kwargs['net']['condition']['observations']

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.action_encoder = nn.Embedding(self.num_agent_actions, self.num_agent_actions)

        self.state_encoder = nn.Sequential(
            init_(nn.Conv2d(self.state_shape[0], 32, 3, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 4 * 4, hidden_size)), nn.ReLU())

        self.mlp = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                                 init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                                 init_(nn.Linear(hidden_size, self.num_agent_actions)))

        # This check is required only to support backward compatibility to pre-trained models
        if (self.setup =='RL' or self.setup =='supervisedRL') and kwargs['rl']['algo']['name'] != 'reinforce':
            self.critic = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.softmax = nn.LogSoftmax(dim=-1)

        self.train()


    def _forward_one_pass(self, inputs, rnn_hxs, masks):
        if self.is_recurrent:
            mlp_inputs, rnn_hxs = self._forward_rnn(inputs, rnn_hxs, masks)

        logits = self.mlp(mlp_inputs)

        value = None
        if (self.setup =='RL' or self.setup =='supervisedRL') and self.rl_algorithm != 'reinforce':
            hidden_critic = self.critic(rnn_hxs)
            value = self.critic_linear(hidden_critic)

        return value, logits, rnn_hxs

    def _env_step(self, states, actions, step):
        states = states.detach().cpu().numpy().astype(np.bool_)
        # C x H x W to H x W x C
        states = np.moveaxis(states,[-1,-2,-3], [-2,-3,-1])
        assert states.shape[-1] == 16
        # karel world expects H x W x C
        if step == 0:
            self._world = Data.from_matrix(states)
        new_states = self._world.step(actions.detach().cpu().numpy())
        new_states = np.moveaxis(new_states,[-1,-2,-3], [-3,-1,-2])
        new_states = torch.tensor(new_states, dtype=torch.float32, device=actions.device)
        return new_states


    def forward(self, s_h, a_h, z, teacher_enforcing=True, eval_actions=None, eval_masks_all=None,
                deterministic=False, evaluate=False):
        """

        :param s_h:
        :param a_h:
        :param z:
        :param teacher_enforcing: True if training in supervised setup or evaluating actions in RL setup
        :param eval_actions:
        :param eval_masks_all:
        :param deterministic:
        :param evaluate: True if setup == RL and evaluating actions, False otherwise
        :return:
        """
        if self.setup == 'supervised':
            assert deterministic == True
        # s_h: B x num_demos_per_program x 1 x C x H x W
        batch_size, num_demos_per_program, demo_len, C, H, W = s_h.shape
        new_batch_size = s_h.shape[0] * s_h.shape[1]
        teacher_enforcing = teacher_enforcing and self.use_teacher_enforcing
        old_states = s_h.squeeze().view(new_batch_size, C, H, W)

        """ get state_embedding of one image per demonstration"""
        state_embeddings = self.state_encoder(s_h[:, :, 0, :, :, :].view(new_batch_size, C, H, W))
        state_embeddings = state_embeddings.view(batch_size, num_demos_per_program, self._hidden_size)
        assert state_embeddings.shape[0] == batch_size and state_embeddings.shape[1] == num_demos_per_program
        state_embeddings = state_embeddings.squeeze()

        """ get intention_embeddings"""
        intention_embedding = z.unsqueeze(1).repeat(1, num_demos_per_program, 1)

        """ get action embeddings for initial actions"""
        actions = (self.num_agent_actions - 1) * torch.ones((batch_size * num_demos_per_program, 1), device=s_h.device,
                                                            dtype=torch.long)

        rnn_hxs = intention_embedding.view(batch_size * num_demos_per_program, self._hidden_size)
        masks = torch.ones((batch_size * num_demos_per_program, 1), device=intention_embedding.device, dtype=torch.bool)
        gru_mask = torch.ones((batch_size * num_demos_per_program, 1), device=intention_embedding.device, dtype=torch.bool)
        assert rnn_hxs.shape[0] == gru_mask.shape[0]
        if self._rnn_type == 'LSTM':
            rnn_hxs = (rnn_hxs, rnn_hxs)
        masks_all = []
        value_all = []
        actions_all = []
        action_logits_all = []
        action_log_probs_all = []
        dist_entropy_all = []
        max_a_h_len = self.max_demo_length-1
        for i in range(self.max_demo_length-1):
            """ get action embeddings and concatenate them with intention and state embeddings """
            action_embeddings = self.action_encoder(actions.view(batch_size, num_demos_per_program))
            inputs = torch.cat((intention_embedding, state_embeddings, action_embeddings), dim=-1)
            inputs = inputs.view(batch_size * num_demos_per_program, -1)

            """ forward pass"""
            value, action_logits, rnn_hxs = self._forward_one_pass(inputs, rnn_hxs, gru_mask)

            """ apply a temporary softmax to get action values to calculate masks """
            if self.setup == 'supervised':
                with torch.no_grad():
                    actions = self.softmax(action_logits).argmax(dim=-1).view(-1, 1)
            elif self.setup == 'RL':
                # define distribution over current logits
                dist = FixedCategorical(logits=action_logits)
                # calculate log probabilities
                if evaluate:
                    assert eval_actions[:, i].shape == actions.squeeze().shape, '{}:{}'.format(eval_actions[:, i].shape,
                                                                                               actions.squeeze().shape)
                    action_log_probs = dist.log_probs(eval_actions[:,i])
                else:
                    # sample actions
                    actions = dist.mode() if deterministic else dist.sample()
                    action_log_probs = dist.log_probs(actions)

                # calculate entropy
                dist_entropy = dist.entropy()
                action_log_probs_all.append(action_log_probs)
                dist_entropy_all.append(dist_entropy.view(-1,1))
            else:
                raise NotImplementedError()

            assert masks.shape == actions.shape
            if not evaluate:
                # NOTE: remove this if check and keep mask update line in case we want to speed up training
                if masks.detach().sum().cpu().item() != 0:
                    masks = masks  * (actions < 5)
                masks_all.append(masks)

            value_all.append(value)
            action_logits_all.append(action_logits)
            actions_all.append(actions)

            """ apply teacher enforcing if ground-truth trajectories available """
            if teacher_enforcing:
                if self.setup == 'supervised':
                    actions = a_h[:, :, i].squeeze().long().view(-1, 1)
                else:
                    actions = eval_actions[:, i].squeeze().long().view(-1, 1)

            """ get the next state embeddings for input to the network"""
            if self.states_source != 'initial_state':
                if teacher_enforcing and self.states_source == 'dataset':
                    new_states = s_h[:, :, i+1, :, :, :].view(s_h.shape[0] * s_h.shape[1], C, H, W)
                else:
                    new_states = self._env_step(old_states, actions, i)
                    assert new_states.shape == (batch_size * num_demos_per_program, C, H, W)

                state_embeddings = self.state_encoder(new_states).view(batch_size, num_demos_per_program,
                                                                         self._hidden_size)
                old_states = new_states

        # unmask first <pad> token
        if not evaluate:
            masks_all = torch.stack(masks_all, dim=1).squeeze()
            first_end_token_idx = torch.sum(masks_all.squeeze(), dim=1)
            _ = list(map(unmask_idx2, zip(masks_all, first_end_token_idx)))

        action_logits_all = torch.stack(action_logits_all, dim=1)
        assert action_logits_all.shape[-1] == 6

        if self.setup == 'RL':
            masks_all = eval_masks_all if evaluate else masks_all
            actions_all = torch.cat(actions_all, dim=1)

            raw_action_log_probs_all = torch.cat(action_log_probs_all, dim=1)
            action_log_probs_all = masked_sum(raw_action_log_probs_all, masks_all, dim=1, keepdim=True)

            raw_dist_entropy_all = torch.cat(dist_entropy_all, dim=1)
            dist_entropy_all = masked_mean(raw_dist_entropy_all, masks_all, dim=tuple(range(len(masks_all.shape))))

            # calculate value for program from token values
            if self.rl_algorithm != 'reinforce':
                if self.value_method == 'mean':
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_all = masked_mean(raw_value_all, masks_all, dim=1, keepdim=True)
                else:
                    # calculate value function from hidden states
                    raw_value_all = torch.cat(value_all, dim=1)
                    value_idx = torch.sum(masks_all, dim=1, keepdim=True) - 1
                    assert len(value_idx.shape) == 2 and value_idx.shape[1] == 1
                    value_all = torch.gather(raw_value_all, 1, value_idx)

                    # this value calculation is just for sanity check
                    with torch.no_grad():
                        value_idx_2 = first_end_token_idx.clamp(max=self.max_program_len - 1).long().reshape(-1, 1)
                        value_all_2 = torch.gather(raw_value_all, 1, value_idx_2)
                        assert torch.sum(value_all != value_all_2) == 0
                    assert value_all.shape[0] == batch_size
            else:
                value_all = torch.zeros_like(action_log_probs_all)

            value_all = value_all.view(batch_size, num_demos_per_program, 1)
            actions_all = actions_all.view(batch_size, num_demos_per_program, self.max_demo_length - 1)
            masks_all = masks_all.view(batch_size, num_demos_per_program, self.max_demo_length - 1)
            action_log_probs_all = action_log_probs_all.view(batch_size, num_demos_per_program, 1)

        else:
            value_all = None

        return value_all, actions_all, action_log_probs_all, action_logits_all, masks_all, dist_entropy_all