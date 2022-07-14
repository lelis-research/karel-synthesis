from dsl.production import Production
from dsl.parser import Parser
from embedding.autoencoder.program_vae import ProgramVAE
from embedding.default_config import config
from karel.vec_env import VecEnv
from gym.spaces import Box
import torch
import numpy as np


PROGRAM = 'DEF run m( WHILE c( frontIsClear c) w( move putMarker w) m)'


if __name__ == '__main__':

    dsl = Production.default_karel_production()

    num_agent_actions = len(dsl.get_actions()) + 1
    config['dsl']['num_agent_actions'] = num_agent_actions

    env = VecEnv(action_space=Box(low=0, high=51, shape=(45,), dtype=np.int16))

    model = ProgramVAE(env, **config)

    params = torch.load('weights/LEAPS/best_valid_params.ptp', map_location=torch.device('cpu'))
    model.load_state_dict(params[0], strict=False)

    input_program_tokens = Parser.tokens_to_list(PROGRAM)
    input_program = torch.tensor(Parser.pad_list(input_program_tokens, 45))
    input_program_len = torch.tensor([len(input_program_tokens)])

    _, rnn_hxs = model.vae.encoder(torch.stack((input_program, input_program)), input_program_len)

    z = model.vae._sample_latent(rnn_hxs.squeeze())

    _, pred_programs_all, pred_programs_len, _, _, _, _, _, _ = model.vae.decoder(
        None, torch.stack((z, z)), teacher_enforcing=False, deterministic=True, evaluate=False
    )

    output_program = Parser.list_to_tokens(pred_programs_all.detach().cpu().numpy().tolist()[0][0:pred_programs_len[0]])

    print('embedding space:', z.detach().cpu().numpy().tolist(), 'shape:', z.shape)
    print('decoded program:', output_program, '; len:', pred_programs_len[0])
