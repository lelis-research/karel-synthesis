import logging
import h5py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dsl.production import Production
from config.config import Config
from logger.stdout_logger import StdoutLogger

def get_exec_data(hdf5_file, program_id, num_agent_actions):
    s_h = np.moveaxis(np.copy(hdf5_file[program_id]['s_h']), [-1,-2,-3], [-3,-1,-2])
    a_h = np.copy(hdf5_file[program_id]['a_h'])
    s_h_len = np.copy(hdf5_file[program_id]['s_h_len'])
    a_h_len = np.copy(hdf5_file[program_id]['a_h_len'])

    # Add no-op actions for empty demonstrations
    for i in range(s_h_len.shape[0]):
        if a_h_len[i] == 0:
            assert s_h_len[i] == 1
            a_h_len[i] += 1
            s_h_len[i] += 1
            s_h[i][1, :, :, :] = s_h[i][0, :, :, :]
            a_h[i][0] = num_agent_actions - 1
    
    return s_h, a_h, a_h_len


def make_dataloaders(datadir, dsl: Production, device: torch.device):
    hdf5_file = h5py.File(os.path.join(datadir, 'data.hdf5'), 'r')
    id_file = open(os.path.join(datadir, 'id.txt'), 'r')
    
    logger = StdoutLogger.get_logger()
    
    num_agent_actions = len(dsl.get_actions()) + 1

    logger.info('Loading programs from karel dataset.')
    program_list = []
    id_list = id_file.readlines()
    for program_id in id_list:
        program_id = program_id.strip()
        program = hdf5_file[program_id]['program'][()]
        exec_data = get_exec_data(hdf5_file, program_id, num_agent_actions)
        if program.shape[0] < Config.data_max_program_len:
            program_list.append((program_id, program, exec_data))
    id_file.close()
    logger.info('Total programs with length <= {}: {}'.format(Config.data_max_program_len, len(program_list)))

    rng = np.random.RandomState(Config.env_seed)
    rng.shuffle(program_list)

    split_idx1 = int(Config.data_ratio_train * len(program_list))
    split_idx2 = int((Config.data_ratio_train + Config.data_ratio_val)*len(program_list))
    train_program_list = program_list[:split_idx1]
    valid_program_list = program_list[split_idx1:split_idx2]
    test_program_list = program_list[split_idx2:]

    train_dataset = ProgramDataset(train_program_list, dsl, device)
    val_dataset = ProgramDataset(valid_program_list, dsl, device)
    test_dataset = ProgramDataset(test_program_list, dsl, device)
    
    torch_rng = torch.Generator().manual_seed(Config.env_seed)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.data_batch_size, shuffle=True, drop_last=True, generator=torch_rng)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.data_batch_size, shuffle=True, drop_last=True, generator=torch_rng)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.data_batch_size, shuffle=True, drop_last=True, generator=torch_rng)
    
    return train_dataloader, val_dataloader, test_dataloader


class ProgramDataset(Dataset):
    def __init__(self, program_list: list, dsl: Production, device: torch.device):
        self.device = device
        self.programs = program_list
        # need this +1 as DEF token is input to decoder, loss will be calculated only from run token
        self.max_program_len = Config.data_max_program_len + 1
        self.max_demo_length = Config.data_max_demo_length
        self.num_program_tokens = len(dsl.get_tokens()) + 1
        self.num_agent_actions = len(dsl.get_actions()) + 1

    def __len__(self):
        return len(self.programs)

    def __getitem__(self, idx):
        _, prog, exec_data = self.programs[idx]

        prog = torch.from_numpy(prog).to(self.device).to(torch.long)
        program_len = prog.shape[0]
        prog_sufix = torch.tensor((self.max_program_len - program_len - 1) * [self.num_program_tokens - 1],
                                  device=self.device, dtype=torch.long)
        prog = torch.cat((prog, prog_sufix))
        
        # load exec data
        s_h, a_h, a_h_len = exec_data
        
        a_h_expanded = np.ones((a_h.shape[0], self.max_demo_length), dtype=int) * (self.num_agent_actions - 1)
        s_h_expanded = np.zeros((s_h.shape[0], self.max_demo_length, *s_h.shape[2:]), dtype=bool)

        # Add no-op actions for empty demonstrations
        for i in range(a_h_len.shape[0]):
            a_h_expanded[i, 1:a_h_len[i]+1] = a_h[i, :a_h_len[i]]
            s_h_expanded[i, :a_h_len[i]+1] = s_h[i, :a_h_len[i]+1]
            s_h_expanded[i, a_h_len[i]+1:] = s_h[i, a_h_len[i]] * (self.max_demo_length - a_h_len[i] + 1)
        
        s_h = torch.tensor(s_h_expanded, device=self.device, dtype=torch.float32)
        a_h = torch.tensor(a_h_expanded, device=self.device, dtype=torch.long)
        
        prog = prog.repeat(a_h.shape[0], 1)

        prog_mask = (prog != self.num_program_tokens - 1)
        a_h_mask = (a_h != self.num_agent_actions - 1)

        return s_h, a_h, a_h_mask, prog, prog_mask