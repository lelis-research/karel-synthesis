import torch
from dsl.parser import Parser
from dsl.production import Production
from embedding.autoencoder.leaps_vae import LeapsVAE
from config.config import Config


PROGRAM = 'DEF run m( WHILE c( leftIsClear c) w( turnRight turnLeft move IFELSE c( markersPresent c) i( pickMarker i) ELSE e( move e) w) m)'


if __name__ == '__main__':

    dsl = Production.default_karel_production()

    device = torch.device('cpu')

    config = Config(hidden_size=64)

    model = LeapsVAE(dsl, device, config)

    params = torch.load('output/leaps_vae/model/best_val.ptp', map_location=device)
    model.load_state_dict(params, strict=False)

    input_program_tokens = Parser.tokens_to_list(PROGRAM)
    input_program = torch.tensor(Parser.pad_list(input_program_tokens, 45))
    
    z = model.encode_program(input_program)

    pred_progs = model.decode_vector(z)

    output_program = Parser.list_to_tokens([0] + pred_progs.detach().cpu().numpy().tolist())

    # print('embedding space:', z.detach().cpu().numpy().tolist(), 'shape:', z.shape)
    print('decoded program:', output_program)
