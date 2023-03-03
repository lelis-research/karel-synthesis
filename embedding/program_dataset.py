import random
import h5py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

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


def make_datasets(datadir, max_program_len, max_demo_length, num_program_tokens, num_agent_actions, device, logger):
    """ Given the path to main dataset, split the data into train, valid, test and create respective pytorch Datasets

    Parameters:
        :param datadir (str): patth to main dataset (should contain 'data.hdf5' and 'id.txt')
        :param max_program_len (int): 
        :param num_program_tokens (int): number of program tokens in karel DSL
        :param num_agent_actions (int): number of actions karel agent can take
        :param device(torch.device): dataset target device: torch.device('cpu') or torch.device('cuda:X')

    Returns:
        :return train_dataset(torch.utils.data.Dataset): training dataset
        :return valid_dataset(torch.utils.data.Dataset): validation dataset
        :return test_dataset(torch.utils.data.Dataset): test dataset

    """
    hdf5_file = h5py.File(os.path.join(datadir, 'data.hdf5'), 'r')
    id_file = open(os.path.join(datadir, 'id.txt'), 'r')

    logger.info('Loading programs from karel dataset.')
    program_list = []
    id_list = id_file.readlines()
    for program_id in id_list:
        program_id = program_id.strip()
        program = hdf5_file[program_id]['program'][()]
        exec_data = get_exec_data(hdf5_file, program_id, num_agent_actions)
        if program.shape[0] < max_program_len:
            program_list.append((program_id, program, exec_data))
    id_file.close()
    logger.info('Total programs with length <= {}: {}'.format(max_program_len, len(program_list)))

    random.shuffle(program_list)

    train_r, val_r, test_r = 0.7, 0.15, 0.15
    split_idx1 = int(train_r*len(program_list))
    split_idx2 = int((train_r+val_r)*len(program_list))
    train_program_list = program_list[:split_idx1]
    valid_program_list = program_list[split_idx1:split_idx2]
    test_program_list = program_list[split_idx2:]

    train_dataset = ProgramDataset(train_program_list, max_program_len, max_demo_length, num_program_tokens, num_agent_actions, device)
    val_dataset = ProgramDataset(valid_program_list, max_program_len, max_demo_length, num_program_tokens, num_agent_actions, device)
    test_dataset = ProgramDataset(test_program_list, max_program_len, max_demo_length, num_program_tokens, num_agent_actions, device)
    return train_dataset, val_dataset, test_dataset


class ProgramDataset(Dataset):
    def __init__(self, program_list, max_program_len, max_demo_length, num_program_tokens, num_agent_actions, device):
        """ Init function for karel program dataset

        Parameters:
            :param program_list (list): list containing information about each program in dataset
            :param config (dict): all configs in dict format
            :param num_program_tokens (int): number of program tokens in karel DSL
            :param num_agent_actions (int): number of actions karel agent can take
            :param device(torch.device): dataset target device: torch.device('cpu') or torch.device('cuda:X')

        Returns: None
        """
        self.device = device
        self.programs = program_list
        # need this +1 as DEF token is input to decoder, loss will be calculated only from run token
        self.max_program_len = max_program_len + 1
        self.max_demo_length = max_demo_length
        self.num_program_tokens = num_program_tokens
        self.num_agent_actions = num_agent_actions

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

        prog_mask = (prog != self.num_program_tokens - 1)
        a_h_mask = (a_h != self.num_agent_actions - 1)

        return s_h, a_h, a_h_mask, prog, prog_mask