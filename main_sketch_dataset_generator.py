import pickle
import numpy as np
import torch

import tqdm
from config import Config
from dsl import DSL
from dsl.program_generator import ProgramGenerator
from karel.world_generator import WorldGenerator
from logger.stdout_logger import StdoutLogger
from search.sketch_sampler import SketchSampler
from multiprocessing import pool
from vae.program_dataset import ProgramDataset, ProgramsAndDemosDataset, ProgramsOnlyDataset


if __name__ == '__main__':

    dsl = DSL.init_default_karel().extend_dsl()
    device = torch.device('cpu')
    
    sketch_sampler = SketchSampler(statements_only=True)
    
    Config.data_max_program_size = 8
    Config.data_max_program_depth = 3
    Config.data_max_program_sequence = 6
    Config.datagen_sketch_iterations = 6
        
    sketches_dataset = []
    programs_and_sketches_dataset = []
        
    program_setup = f'size{Config.data_max_program_size}_dep{Config.data_max_program_depth}_seq{Config.data_max_program_sequence}'
    sketch_setup = f'sk{Config.datagen_sketch_iterations}'
    
    Config.data_program_dataset_path = f'data/programs_{program_setup}_only.pkl'
    Config.data_class_name = 'ProgramsOnlyDataset'
    
    with open(Config.data_program_dataset_path, 'rb') as f:
        program_list = pickle.load(f)
    
    data_cls = globals()[Config.data_class_name]
    
    dataset = data_cls(program_list, dsl, device)
    
    programs_nodes = []
    program_dataset = []
    
    for item in dataset:
        _, _, _, prog, prog_mask = item
        prog_token_len = prog_mask.sum().item()
        program_dataset.append(prog[:prog_token_len].tolist())
        prog_str = dsl.parse_int_to_str(prog[:prog_token_len].tolist())
        prog_nodes = dsl.parse_str_to_node(prog_str)
        programs_nodes.append(prog_nodes)
    
    StdoutLogger.log('Generator', f'Generating sketches in setup {sketch_setup}')
    
    def sample(program_nodes):
        s = sketch_sampler.sample_sketch(program_nodes)
        return s
    
    with pool.Pool() as pl:
        sketches_nodes = list(tqdm.tqdm(pl.imap(sample, programs_nodes), total=len(programs_nodes)))
    
    # sketches_nodes = [sample(p) for p in programs_nodes]
    
    for p, sketch_nodes in zip(program_dataset, sketches_nodes):
        
        s = dsl.parse_node_to_int(sketch_nodes)
        
        sketches_dataset.append(s)
        
        programs_and_sketches_dataset.append((p, s))
    
    StdoutLogger.log('Generator', 'Saving files.')

    with open(f'data/sketches_{program_setup}_{sketch_setup}.pkl', 'wb') as f:
        pickle.dump(sketches_dataset, f)
    with open(f'data/programs_{program_setup}_and_sketches_{sketch_setup}.pkl', 'wb') as f:
        pickle.dump(programs_and_sketches_dataset, f)
