from __future__ import annotations
import numpy as np
import pickle

from dsl import DSL
from dsl.parser import Parser
from vae.program_dataset import load_programs
from search.sketch_sampler import SketchSampler


if __name__ == '__main__':
    
    dsl = DSL.init_default_karel()
    
    program_list = load_programs(dsl)
    
    sketch_sampler = SketchSampler()

    sketches_list = []

    for program_info in program_list:
        program_nodes = dsl.parse_int_to_node(program_info[1])

        sketch = sketch_sampler.sample_sketch(program_nodes, 4)

        sketch_str = dsl.parse_node_to_str(sketch)
        
        sketch_tokens = dsl.parse_str_to_int(sketch_str)
        
        sketches_list.append((program_info[0], program_info[1], program_info[2], np.array(sketch_tokens)))
    
    with open('data/sketches.pkl', 'wb') as f:
        pickle.dump(sketches_list, f)
    