import h5py
from torch.utils.data import Dataset


class ProgramLoader(Dataset):

    def __init__(self, data_path: str):
        self.data_path = data_path

    def __getitem__(self, idx):
        with h5py.File(self.data_path, 'r') as hf:
            z = hf[f'{idx:06d}']['z'][:]
            s_s = hf[f'{idx:06d}']['s_s'][:]
            s_f = hf[f'{idx:06d}']['s_f'][:]
            # TODO: crunch channels 0-3 of s_f into one and remove other channels
            return z, s_s, s_f
