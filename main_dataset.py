from __future__ import annotations
import pickle

from dsl import DSL
from vae.program_dataset import load_programs


if __name__ == '__main__':
    
    dsl = DSL.init_default_karel()
    
    program_list = load_programs(dsl)
    
    with open('data/programs.pkl', 'wb') as f:
        pickle.dump(program_list, f)