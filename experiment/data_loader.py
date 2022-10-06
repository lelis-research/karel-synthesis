import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class ProgramDataset(Dataset):

    def __init__(self, data_path: str):
        data = np.load(data_path)
        self.data_z = torch.tensor(data['z'], dtype=torch.float)
        self.data_s_s = torch.tensor(data['s_s'], dtype=torch.float)
        self.data_s_s = torch.moveaxis(self.data_s_s, [-1,-2,-3], [-3,-1,-2])
        self.data_s_f = torch.tensor(data['s_f'], dtype=torch.float)
        self.data_s_f = self.data_s_f[:,:,:,0:4]
        self.data_s_f = torch.sum(self.data_s_f, dim=3)
        self.data_s_f = torch.flatten(self.data_s_f, start_dim=1)
        self.data_s_f = torch.argmax(self.data_s_f, dim=1)
        pass

    def __getitem__(self, idx):
        return self.data_z[idx], self.data_s_s[idx], self.data_s_f[idx]

    def __len__(self):
        return len(self.data_z)
