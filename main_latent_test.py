import torch
from dsl.parser import Parser
from dsl.production import Production
from vae.models.leaps_vae import LeapsVAE
from config.config import Config


PROGRAM = 'DEF run m( WHILE c( rightIsClear c) w( move w) m)'


if __name__ == '__main__':

    Config.model_hidden_size = 32

    dsl = Production.default_karel_production()

    device = torch.device('cpu')

    model = LeapsVAE(dsl, device)

    params = torch.load('output/leaps_vae_debug/model/best_val.ptp', map_location=device)
    model.load_state_dict(params, strict=False)

    input_program_tokens = Parser.str_to_tokens(PROGRAM)
    input_program = torch.tensor(Parser.pad_tokens(input_program_tokens, 45))
    
    input_program = torch.stack((input_program, input_program))
    
    z = model.encode_program(input_program)

    pred_progs = model.decode_vector(z)

    output_program = Parser.tokens_to_str(pred_progs[0])

    # print('latent vector:', z.detach().cpu().numpy().tolist(), 'shape:', z.shape)
    print('decoded program:', output_program)
