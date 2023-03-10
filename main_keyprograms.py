from dsl.production import Production
from dsl.parser import Parser
from embedding.autoencoder.program_vae import ProgramVAE
from config.config import Config
import torch
import pickle


PROGRAMS = [
    'DEF run m( IFELSE c( frontIsClear c) i( move i) ELSE e( turnRight e) m)',
    'DEF run m( REPEAT R=4 r( move turnRight r) m)',
    'DEF run m( WHILE c( frontIsClear c) w( move w) m)',
    'DEF run m( WHILE c( rightIsClear c) w( turnRight IFELSE c( markersPresent c) i( pickMarker i) ELSE e( move e) w) m)',
    'DEF run m( IFELSE c( noMarkersPresent c) i( putMarker WHILE c( not c( frontIsClear c) c) w( turnRight w) i) ELSE e( move e) m)',
    'DEF run m( REPEAT R=2 r( IFELSE c( markersPresent c) i( pickMarker i) ELSE e( move turnLeft e) r) m)',
    'DEF run m( IFELSE c( frontIsClear c) i( move IF c( noMarkersPresent c) i( putMarker i) i) ELSE e( turnLeft e) m)',
    'DEF run m( WHILE c( leftIsClear c) w( turnRight WHILE c( markersPresent c) w( pickMarker w) w) m)'
]


if __name__ == '__main__':

    dsl = Production.default_karel_production()

    device = torch.device('cpu')
    
    for size in [8, 16, 32, 64, 128, 256]:
        
        print(f'model size: {size}')

        config = Config(hidden_size=size)

        model = ProgramVAE(dsl, device)

        params = torch.load(f'../leaps/weights/leapspl_{size}.ptp', map_location=torch.device('cpu'))
        model.load_state_dict(params[0], strict=False)
        
        zs = []
        
        for i, p in enumerate(PROGRAMS):

            input_program_tokens = Parser.tokens_to_list(p)
            input_program = torch.tensor(Parser.pad_list(input_program_tokens, 45))
            input_program_len = torch.tensor([len(input_program_tokens)])

            _, rnn_hxs = model.vae.encoder(torch.stack((input_program, input_program)), input_program_len)

            z = model.vae._sample_latent(rnn_hxs.squeeze())
            zs.append(z)
            
            with open(f'key_programs/leapspl{size}_kp{i}.pkl', 'wb') as f:
                pickle.dump(z, f)

            pred_programs_all, pred_programs_len, _, _, _, _, _, _ = model.vae.decoder(
                None, torch.stack((z, z)), teacher_enforcing=False, deterministic=True, evaluate=False
            )

            output_program = Parser.list_to_tokens([0] + pred_programs_all.detach().cpu().numpy().tolist()[0][0:pred_programs_len[0]-1])

            # print('embedding space:', z.detach().cpu().numpy().tolist(), 'shape:', z.shape)
            print(output_program)
            print(output_program == p)
        
        zs = torch.stack(zs)
        
        with open(f'key_programs/leapspl{size}.pkl', 'wb') as f:
            pickle.dump(zs, f)
