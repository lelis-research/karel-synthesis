import h5py
import sys


sys.path.insert(0, '.')

from karel.world import World
from dsl.parser import Parser

if __name__ == '__main__':

    with h5py.File('data/experiment_backup.hdf5', 'r') as hf:

        print('Lines in data:', len(hf.keys()))

        idx = 0

        print(f'Sample in dataset: [idx == {idx:06d}]\n')

        print('Raw data:')

        z = hf[f'{idx:06d}']['z'][:]

        print('prog:', hf[f'{idx:06d}']['prog'][:])
        print('z:',    hf[f'{idx:06d}']['z'][:])
        print('s_s:',  hf[f'{idx:06d}']['s_s'][:])
        print('s_f:',  hf[f'{idx:06d}']['s_f'][:])

        print('\nShape of raw data:')

        print('prog:', hf[f'{idx:06d}']['prog'][:].shape)
        print('z:',    hf[f'{idx:06d}']['z'][:].shape)
        print('s_s:',  hf[f'{idx:06d}']['s_s'][:].shape)
        print('s_f:',  hf[f'{idx:06d}']['s_f'][:].shape)
        
        print('\nInterpreted data:\n')

        print('Parsed program:', Parser.list_to_tokens(hf[f'{idx:06d}']['prog']).replace(' <pad>', ''))
        print('\nStart state plot:')
        print(World(hf[f'{idx:06d}']['s_s'][:]).to_string())
        print('\n\nFinal state plot:')
        print(World(hf[f'{idx:06d}']['s_f'][:]).to_string())
