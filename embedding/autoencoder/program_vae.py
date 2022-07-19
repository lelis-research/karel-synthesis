import torch
import torch.nn as nn
from dsl.production import Production
from embedding.autoencoder.condition_policy import ConditionPolicy
from embedding.autoencoder.vae import VAE
from embedding.config.config import Config


class ProgramVAE(nn.Module):
    def __init__(self, dsl: Production, config: Config):
        super(ProgramVAE, self).__init__()
        num_outputs = len(dsl.get_tokens()) + 1
        self.num_program_tokens = num_outputs
        # two_head policy shouldn't have <pad> token in action distribution, but syntax checker forces it
        # even if its included, <pad> will always have masked probability = 0, so implementation vise it should be fine

        self.num_demo_per_program = config.num_demo_per_program
        self.max_demo_length = config.max_demo_length

        self.teacher_enforcing = config.use_teacher_enforcing
        self.vae = VAE(num_outputs, dsl, config)
        self.condition_policy = ConditionPolicy(dsl, config)

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.vae.latent_dim

    def forward(self, programs, program_masks, init_states, a_h, rnn_hxs=None, masks=None, action=None, output_mask_all=None, eop_action=None,
                agent_actions=None, agent_action_masks=None, deterministic=False, evaluate=False, reinforce_step=False):

        if not reinforce_step:
            output, z = self.vae(programs, program_masks, self.teacher_enforcing, deterministic=deterministic, reinforce_step=reinforce_step)
            pred_programs, pred_programs_len, _, output_logits, eop_pred_programs, eop_output_logits, pred_program_masks, _ = output
            _, _, action_logits, action_masks, _ = self.condition_policy(init_states, a_h, z, self.teacher_enforcing,
                                                                         deterministic=deterministic, reinforce_step=reinforce_step)
            return pred_programs, pred_programs_len, output_logits, eop_pred_programs, eop_output_logits, \
                   pred_program_masks, action_logits, action_masks, z

        # TODO: isn't the lines below the same as the above if? check

        # output, z = self.vae(programs, program_masks, self.teacher_enforcing)
        """ VAE forward pass """
        program_lens = program_masks.squeeze().sum(dim=-1)
        _, h_enc = self.vae.encoder(programs, program_lens)
        z = self.vae._sample_latent(h_enc.squeeze())

        """ decoder forward pass """
        output = self.vae.decoder(programs, z, teacher_enforcing=evaluate, action=action,
                                  output_mask_all=output_mask_all, eop_action=eop_action, deterministic=deterministic,
                                  evaluate=evaluate, reinforce_step=reinforce_step)

        pred_programs, pred_programs_len, pred_programs_log_probs, output_logits, eop_pred_programs,\
        eop_output_logits, pred_program_masks, dist_entropy = output

        """ Condition policy rollout using sampled latent vector """
        if reinforce_step:
            agent_actions, agent_action_log_probs, _, agent_action_masks, \
            agent_action_dist_entropy = self.condition_policy(init_states, None, z,
                                                                          teacher_enforcing=evaluate,
                                                                          eval_actions=agent_actions,
                                                                          eval_masks_all=agent_action_masks,
                                                                          deterministic=deterministic,
                                                                          evaluate=evaluate,
                                                                          reinforce_step=reinforce_step)
        else:
            batch_size = z.shape[0]
            agent_actions = torch.zeros((batch_size, self.num_demo_per_program, self.max_demo_length - 1), device=z.device, dtype=torch.long)
            agent_action_log_probs = torch.zeros((batch_size, self.num_demo_per_program, 1), device=z.device, dtype=torch.float)
            agent_action_masks = torch.zeros((batch_size, self.num_demo_per_program, self.max_demo_length - 1), device=z.device, dtype=torch.bool)
            agent_action_dist_entropy = torch.zeros(1, device=z.device, dtype=torch.float)


        """ calculate latent log probs """
        distribution_params = torch.stack((self.vae.z_mean, self.vae.z_sigma), dim=1)
        latent_log_probs, latent_dist_entropy = pred_programs_log_probs, dist_entropy

        return pred_programs, pred_programs_log_probs, z, pred_program_masks, eop_pred_programs,\
                agent_actions, agent_action_log_probs, agent_action_masks, z, distribution_params, dist_entropy,\
                agent_action_dist_entropy, latent_log_probs, latent_dist_entropy

    def act(self, programs, rnn_hxs, masks, init_states, deterministic=False, reinforce_step=False):
        program_masks = programs != self.num_program_tokens-1
        outputs = self(programs.long(), program_masks, init_states, None, rnn_hxs, masks, deterministic=deterministic,
                       evaluate=False, reinforce_step=reinforce_step)
        return outputs

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, output_mask_all, eop_action, agent_actions,
                         agent_action_masks, program_ids, deterministic=False):
        programs, init_states, z = inputs
        program_masks = programs != self.num_program_tokens - 1

        outputs = self(programs.long(), program_masks, init_states, None, rnn_hxs=rnn_hxs, masks=masks,
                       action=action.long(), output_mask_all=output_mask_all, eop_action=eop_action,
                       agent_actions=agent_actions, agent_action_masks=agent_action_masks,
                       deterministic=deterministic, evaluate=True)
        _, pred_programs_log_probs, z, pred_program_masks, _, _, agent_action_log_probs, \
        _, _, distribution_params, dist_entropy, agent_action_dist_entropy, latent_log_probs, \
        latent_dist_entropy = outputs

        return pred_programs_log_probs, dist_entropy, z, pred_program_masks, agent_action_log_probs, \
               agent_action_dist_entropy, distribution_params, latent_log_probs, latent_dist_entropy
