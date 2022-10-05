import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ProgramDataset(Dataset):

    def __init__(self, data_path: str):
        self.data_path = data_path

    def __getitem__(self, idx):
        with h5py.File(self.data_path, 'r') as hf:
            z = hf[f'{idx:06d}']['z'][:]
            s_s = hf[f'{idx:06d}']['s_s'][:]
            s_f = np.sum(hf[f'{idx:06d}']['s_f'][:,:,0:4], axis=-1)
            z = torch.tensor(z, dtype=torch.float)
            s_s = torch.moveaxis(torch.tensor(s_s, dtype=torch.float), [-1,-2,-3], [-3,-1,-2])
            s_f = torch.flatten(torch.tensor(s_f, dtype=torch.float))
            return z, s_s, s_f

    def __len__(self):
        with h5py.File(self.data_path, 'r') as hf:
            return len(hf.keys())
