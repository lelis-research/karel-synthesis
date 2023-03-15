from __future__ import annotations
import pickle

from dsl.production import Production
from embedding.program_dataset import load_programs


if __name__ == '__main__':
    
    dsl = Production.default_karel_production()
    
    program_list = load_programs(dsl)
    
    with open('data/programs.pkl', 'wb') as f:
        pickle.dump(program_list, f)