import pickle
import numpy as np

import tqdm
from config import Config
from dsl import DSL
from dsl.generator import ProgramGenerator
from karel.world_generator import WorldGenerator
from logger.stdout_logger import StdoutLogger
from search.sketch_sampler import SketchSampler
from multiprocessing import pool

if __name__ == '__main__':

    dsl = DSL.init_default_karel()
    program_generator = ProgramGenerator(dsl)
    sketch_sampler = SketchSampler()
    world_generator = WorldGenerator()
    
    generate_demos = Config.datagen_generate_demos
    generate_sketches = Config.datagen_generate_sketches
    
    seen_programs = set()
    program_dataset = []
    program_and_demos_dataset = []
    sketches_dataset = []
    programs_and_sketches_dataset = []
    
    program_setup = f'l{Config.data_max_program_len}_d{Config.datagen_max_depth}_s{Config.datagen_max_sequential_length}'
    sketch_setup = f'n{Config.datagen_sketch_iterations}'
    
    StdoutLogger.log('Generator', f'Generating programs in setup {program_setup}')
    
    with tqdm.tqdm(total=Config.datagen_num_programs) as pbar:

        while len(program_dataset) < Config.datagen_num_programs:
            program = program_generator.generate_program()
            
            program_str = dsl.parse_node_to_str(program)
            if program_str in seen_programs: continue
            
            if generate_demos:
                try:
                    s_h, a_h = program_generator.generate_demos(program, world_generator,
                                                                Config.data_num_demo_per_program,
                                                                Config.data_max_demo_length)
                except Exception: # In case a timeout occurs
                    continue
            
            seen_programs.add(program_str)
            
            p = dsl.parse_str_to_int(program_str)
            
            program_dataset.append(p)
            
            if generate_demos:
                program_and_demos_dataset.append((p, s_h, a_h))
            
            pbar.update(1)
        
    if generate_sketches:
        programs_nodes = [dsl.parse_int_to_node(p) for p in program_dataset]
        
        StdoutLogger.log('Generator', f'Generating sketches in setup {sketch_setup}')
        
        with tqdm.tqdm(total=Config.datagen_num_programs) as pbar:
            def sample(program_nodes):
                s = sketch_sampler.sample_sketch(program_nodes)
                pbar.update(1)
                return s
            
            with pool.Pool() as pl:
                sketches_nodes = pl.map(sample, programs_nodes)
        
        for p, sketch_nodes in zip(program_dataset, sketches_nodes):
            
            s = dsl.parse_node_to_int(sketch_nodes)
            
            sketches_dataset.append(s)
            
            programs_and_sketches_dataset.append((p, s))
    
    StdoutLogger.log('Generator', 'Saving files.')
    
    with open(f'data/programs_{program_setup}_only.pkl', 'wb') as f:
        pickle.dump(program_dataset, f)
    if generate_demos:
        with open(f'data/programs_{program_setup}_and_demos.pkl', 'wb') as f:
            pickle.dump(program_and_demos_dataset, f)
    if generate_sketches:
        with open(f'data/sketches_{program_setup}_{sketch_setup}.pkl', 'wb') as f:
            pickle.dump(sketches_dataset, f)
        with open(f'data/programs_{program_setup}_and_sketches_{sketch_setup}.pkl', 'wb') as f:
            pickle.dump(programs_and_sketches_dataset, f)