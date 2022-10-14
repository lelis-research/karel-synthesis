from datetime import datetime
import logging
import h5py
import numpy as np
import tqdm
import sys
import torch
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, '.')

from dsl.production import Production
from dsl.parser import Parser
from embedding.autoencoder.program_vae import ProgramVAE
from embedding.config.config import Config
from embedding.program_dataset import make_datasets
from karel.world import World


if __name__ == '__main__':

    dsl = Production.default_karel_production()

    device = torch.device('cpu')

    config = Config(hidden_size=256)

    model = ProgramVAE(dsl, device, config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_handlers = [
        logging.StreamHandler(sys.stdout)
    ]
    logging.basicConfig(handlers=log_handlers, format='%(asctime)s: %(message)s', level=logging.DEBUG)
    logger = logging.getLogger('trainer')

    params = torch.load('weights/LEAPS/best_valid_params.ptp', map_location=torch.device('cpu'))
    model.load_state_dict(params[0], strict=False)

    p_train_dataset, p_val_dataset, p_test_dataset = make_datasets(
        'data/program_dataset', config.max_program_len, config.max_demo_length, 
        model.num_program_tokens, len(dsl.get_actions()) + 1, device, logger)

    concat_dataset = ConcatDataset([p_train_dataset, p_val_dataset, p_test_dataset])

    p_dataloader = DataLoader(concat_dataset, batch_size=2, shuffle=True, drop_last=True)

    # idx = 0

    data_z = []
    data_s_s = []
    data_s_f = []
    data_prog = []

    # with h5py.File('data/experiment.hdf5', 'w') as f:

    for batch in tqdm.tqdm(p_dataloader):

        prog_batch, _, trg_mask, s_batch, a_batch, _ = batch
        output_batch = model(prog_batch, trg_mask, s_batch, a_batch, deterministic=True)
        out_prog_batch, _, _, _, _, _, _, _, z_batch = output_batch

        inp_prog_arr = prog_batch.detach().cpu().numpy().tolist()
        out_prog_arr = out_prog_batch.detach().cpu().numpy().tolist()
        a_nparr = a_batch.detach().cpu().numpy()
        s_nparr = s_batch.detach().cpu().numpy()
        z_nparr = z_batch.detach().cpu().numpy()

        for inp_prog, out_prog, s, a, z in zip(
                inp_prog_arr, out_prog_arr, s_nparr, a_nparr, z_nparr
            ):

            s = np.moveaxis(s.squeeze(), [-4, -1, -2, -3], [-4, -2, -3, -1]).astype(bool)

            inp_prog_str = Parser.list_to_tokens(inp_prog).replace(' <pad>', '')
            # out_prog_str = Parser.list_to_tokens(out_prog).replace(' <pad>', '')

            for s_states, actions in zip(s, a):

                world = World(s_states)

                # TODO: get trajectory

                for a in actions:
                    if a == 5: break
                    world.run_action(a)

                f_states = world.get_state()

                data_z.append(z)
                data_s_s.append(s_states)
                data_s_f.append(f_states)
                data_prog.append(inp_prog)

                # h5py_grp = f.create_group(f'{idx:06d}')
                # idx = idx + 1

                # h5py_grp['z'] = z
                # h5py_grp['s_s'] = s_states
                # h5py_grp['s_f'] = f_states
                # h5py_grp['prog'] = inp_prog

    # np.save('data/experiment.npy', data)
    np.savez(
        'data/experiment.npz',
        z=np.array(data_z), s_s=np.array(data_s_s),
        s_f=np.array(data_s_f), prog=np.array(data_prog)
    )
