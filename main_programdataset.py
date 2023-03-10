from datetime import datetime
import logging
import h5py
import numpy as np
import pickle
import tqdm
import sys
import torch
from torch.utils.data import DataLoader, ConcatDataset
from dsl.production import Production
from dsl.parser import Parser
from embedding.autoencoder.program_vae import ProgramVAE
from config.config import Config
from embedding.program_dataset import make_datasets
from karel.world import World


if __name__ == '__main__':

    dsl = Production.default_karel_production()

    device = torch.device('cpu')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_handlers = [
        logging.StreamHandler(sys.stdout)
    ]
    logging.basicConfig(handlers=log_handlers, format='%(asctime)s: %(message)s', level=logging.DEBUG)
    logger = logging.getLogger('trainer')
    
    config = Config()

    model = ProgramVAE(dsl, device)

    p_train_dataset, p_val_dataset, p_test_dataset = make_datasets(
        'data/program_dataset', Config.max_program_len, Config.max_demo_length, 
        model.num_program_tokens, len(dsl.get_actions()) + 1, device, logger)

    concat_dataset = ConcatDataset([p_train_dataset, p_val_dataset, p_test_dataset])

    p_dataloader = DataLoader(concat_dataset, batch_size=16, shuffle=True, drop_last=True)

    for size in [8, 16, 32, 64, 128, 256]:
        
        print(f'model size: {size}')

        config = Config(hidden_size=size)

        model = ProgramVAE(dsl, device)

        params = torch.load(f'../leaps/weights/leapspl_{size}.ptp', map_location=torch.device('cpu'))
        model.load_state_dict(params[0], strict=False)


        z = []

        for batch in tqdm.tqdm(p_dataloader):

            prog_batch, _, trg_mask, s_batch, a_batch, _ = batch
            output_batch = model(prog_batch, trg_mask, s_batch, a_batch, deterministic=True)
            out_prog_batch, _, _, _, _, _, _, _, z_batch = output_batch
            
            z_batch_npy = z_batch.detach().cpu().numpy()
            [z.append(zi) for zi in z_batch_npy]
        
        z = torch.stack([torch.tensor(zi) for zi in z])

        with open(f'all_programs/leapspl{size}.pkl', 'wb') as f:
            pickle.dump(z, f)