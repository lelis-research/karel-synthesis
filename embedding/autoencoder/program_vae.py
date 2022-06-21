import torch
import torch.nn as nn
from embedding.autoencoder.condition_policy import ConditionPolicy
from embedding.autoencoder.vae import VAE


class ProgramVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ProgramVAE, self).__init__()
        envs = args[0]
        action_space = envs.action_space
        num_outputs = int(action_space.high[0]) if not kwargs['two_head'] else int(action_space.high[0] - 1)
        num_program_tokens = num_outputs if not kwargs['two_head'] else num_outputs + 1
        # two_head policy shouldn't have <pad> token in action distribution, but syntax checker forces it
        # even if its included, <pad> will always have masked probability = 0, so implementation vise it should be fine
        if kwargs['two_head'] and kwargs['grammar'] == 'handwritten':
            num_outputs = int(action_space.high[0])

        self._tanh_after_sample = kwargs['net']['tanh_after_sample']
        self._debug = kwargs['debug']
        self.use_decoder_dist = kwargs['net']['controller']['use_decoder_dist']
        self.use_condition_policy_in_rl = kwargs['rl']['loss']['condition_rl_loss_coef'] > 0.0
        self.num_demo_per_program = kwargs['rl']['envs']['executable']['num_demo_per_program']
        self.max_demo_length = kwargs['rl']['envs']['executable']['max_demo_length']

        self.num_program_tokens = num_program_tokens
        self.teacher_enforcing = kwargs['net']['decoder']['use_teacher_enforcing']
        self.vae = VAE(num_outputs, num_program_tokens, **kwargs)
        self.condition_policy = ConditionPolicy(envs, **kwargs)

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.vae.latent_dim

    @property
    def is_recurrent(self):
        return self.vae.encoder.is_recurrent

    def forward(self, programs, program_masks, init_states, a_h, rnn_hxs=None, masks=None, action=None, output_mask_all=None, eop_action=None,
                agent_actions=None, agent_action_masks=None, deterministic=False, evaluate=False):

        if self.vae.decoder.setup == 'supervised':
            output, z = self.vae(programs, program_masks, self.teacher_enforcing, deterministic=deterministic)
            _, pred_programs, pred_programs_len, _, output_logits, eop_pred_programs, eop_output_logits, pred_program_masks, _ = output
            _, _, _, action_logits, action_masks, _ = self.condition_policy(init_states, a_h, z, self.teacher_enforcing,
                                                                         deterministic=deterministic)
            return pred_programs, pred_programs_len, output_logits, eop_pred_programs, eop_output_logits, \
                   pred_program_masks, action_logits, action_masks, z

        # output, z = self.vae(programs, program_masks, self.teacher_enforcing)
        """ VAE forward pass """
        program_lens = program_masks.squeeze().sum(dim=-1)
        _, h_enc = self.vae.encoder(programs, program_lens)
        z = h_enc.squeeze() if self.vae._vanilla_ae else self.vae._sample_latent(h_enc.squeeze())
        pre_tanh_value = None
        if self._tanh_after_sample or not self.use_decoder_dist:
            pre_tanh_value = z
            z = self.program_vae.vae.tanh(z)

        """ decoder forward pass """
        output = self.vae.decoder(programs, z, teacher_enforcing=evaluate, action=action,
                                  output_mask_all=output_mask_all, eop_action=eop_action, deterministic=deterministic,
                                  evaluate=evaluate)

        value, pred_programs, pred_programs_len, pred_programs_log_probs, output_logits, eop_pred_programs,\
        eop_output_logits, pred_program_masks, dist_entropy = output

        """ Condition policy rollout using sampled latent vector """
        if self.condition_policy.setup == 'RL' and self.use_condition_policy_in_rl:
            agent_value, agent_actions, agent_action_log_probs, agent_action_logits, agent_action_masks, \
            agent_action_dist_entropy = self.condition_policy(init_states, None, z,
                                                                          teacher_enforcing=evaluate,
                                                                          eval_actions=agent_actions,
                                                                          eval_masks_all=agent_action_masks,
                                                                          deterministic=deterministic,
                                                                          evaluate=evaluate)
        else:
            batch_size = z.shape[0]
            agent_value = torch.zeros((batch_size, self.num_demo_per_program, 1), device=z.device, dtype=torch.long)
            agent_actions = torch.zeros((batch_size, self.num_demo_per_program, self.max_demo_length - 1), device=z.device, dtype=torch.long)
            agent_action_log_probs = torch.zeros((batch_size, self.num_demo_per_program, 1), device=z.device, dtype=torch.float)
            agent_action_masks = torch.zeros((batch_size, self.num_demo_per_program, self.max_demo_length - 1), device=z.device, dtype=torch.bool)
            agent_action_dist_entropy = torch.zeros(1, device=z.device, dtype=torch.float)


        """ calculate latent log probs """
        distribution_params = torch.stack((self.vae.z_mean, self.vae.z_sigma), dim=1)
        if not self.use_decoder_dist:
            latent_log_probs = self.vae.dist.log_probs(z, pre_tanh_value)
            latent_dist_entropy = self.vae.dist.normal.entropy().mean()
        else:
            latent_log_probs, latent_dist_entropy = pred_programs_log_probs, dist_entropy

        return value, pred_programs, pred_programs_log_probs, z, pred_program_masks, eop_pred_programs,\
                agent_value, agent_actions, agent_action_log_probs, agent_action_masks, z, distribution_params, dist_entropy,\
                agent_action_dist_entropy, latent_log_probs, latent_dist_entropy

    def _debug_rl_pipeline(self, debug_input):
        for i, idx in enumerate(debug_input['ids']):
            current_update = "_".join(idx.split('_')[:-1])
            for key in debug_input.keys():
                program_idx = int(idx.split('_')[-1])
                act_program_info = self._debug['act'][current_update][key][program_idx]
                if key == 'ids':
                    assert (act_program_info == debug_input[key][i])
                elif 'agent' in key:
                    assert (act_program_info == debug_input[key].view(-1,act_program_info.shape[0] ,debug_input[key].shape[-1])[i]).all()
                else:
                    assert (act_program_info == debug_input[key][i]).all()

    def act(self, programs, rnn_hxs, masks, init_states, deterministic=False):
        program_masks = programs != self.num_program_tokens-1
        outputs = self(programs.long(), program_masks, init_states, None, rnn_hxs, masks, deterministic=deterministic,
                       evaluate=False)
        return outputs

    def get_value(self, programs, rnn_hxs, masks, init_states, deterministic=False):
        program_masks = programs != self.num_program_tokens-1
        outputs = self(programs.long(), program_masks, init_states, None, rnn_hxs, masks, deterministic=deterministic,
                       evaluate=False)

        value, pred_programs, pred_programs_log_probs, z, pred_program_masks, eop_pred_programs, \
        agent_value, agent_actions, agent_action_log_probs, agent_action_masks, z, distribution_params, dist_entropy, \
        agent_action_dist_entropy, latent_log_probs, latent_dist_entropy = outputs
        return value, agent_value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, output_mask_all, eop_action, agent_actions,
                         agent_action_masks, program_ids, deterministic=False):
        programs, init_states, z = inputs
        program_masks = programs != self.num_program_tokens - 1

        if self._debug:
            self._debug_rl_pipeline({'pred_programs': action,
                                     'pred_program_masks': output_mask_all,
                                     'agent_actions': agent_actions,
                                     'agent_action_masks': agent_action_masks,
                                     'ids': program_ids})

        outputs = self(programs.long(), program_masks, init_states, None, rnn_hxs=rnn_hxs, masks=masks,
                       action=action.long(), output_mask_all=output_mask_all, eop_action=eop_action,
                       agent_actions=agent_actions, agent_action_masks=agent_action_masks,
                       deterministic=deterministic, evaluate=True)
        value, _, pred_programs_log_probs, z, pred_program_masks, _, agent_value, _, agent_action_log_probs, \
        _, _, distribution_params, dist_entropy, agent_action_dist_entropy, latent_log_probs, \
        latent_dist_entropy = outputs

        return value, pred_programs_log_probs, dist_entropy, z, pred_program_masks, agent_value, agent_action_log_probs, \
               agent_action_dist_entropy, distribution_params, latent_log_probs, latent_dist_entropy
