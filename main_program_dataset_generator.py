import pickle

import tqdm
from config import Config
from dsl import DSL
from dsl.generator import ProgramGenerator
from karel.world_generator import WorldGenerator
from search.sketch_sampler import SketchSampler

if __name__ == '__main__':

    Config.datagen_max_depth = 4
    Config.datagen_max_sequential_length = 6
    Config.data_max_program_len = 45
    Config.dsl_include_hole = True

    dsl = DSL.init_default_karel()
    program_generator = ProgramGenerator(dsl)
    sketch_sampler = SketchSampler()
    world_generator = WorldGenerator()
    
    seen_programs = set()
    program_dataset = []
    sketches_dataset = []
    programs_and_sketches_dataset = []
    
    with tqdm.tqdm(total=Config.datagen_num_programs) as pbar:

        while len(program_dataset) < Config.datagen_num_programs:
            program = program_generator.generate_program()
            
            program_str = dsl.parse_node_to_str(program)
            if program_str in seen_programs: continue
            seen_programs.add(program_str)
            
            p = dsl.parse_str_to_int(program_str)
            
            program_dataset.append(p)
            
            sketch_nodes = sketch_sampler.sample_sketch(program, 4)
            s = dsl.parse_node_to_int(sketch_nodes)
            
            sketches_dataset.append(s)
            
            programs_and_sketches_dataset.append((p, s))
            
            pbar.update(1)
        
    with open('data/programs_only.pkl', 'wb') as f:
        pickle.dump(program_dataset, f)
    with open('data/sketches_only.pkl', 'wb') as f:
        pickle.dump(sketches_dataset, f)
    with open('data/programs_and_sketches.pkl', 'wb') as f:
        pickle.dump(programs_and_sketches_dataset, f)