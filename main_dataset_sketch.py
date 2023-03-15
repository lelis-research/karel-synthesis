from __future__ import annotations
import numpy as np
import pickle
from tqdm import tqdm

from dsl.production import Production
from dsl.parser import Parser
from embedding.program_dataset import load_programs
from search.sketch_sampler import SketchSampler


if __name__ == '__main__':
    
    dsl = Production.default_karel_production()
    
    program_list = load_programs(dsl)

    sketches_list = []

    for program_info in tqdm(program_list):
        program_str = Parser.list_to_tokens(program_info[1])

        program_nodes = Parser.tokens_to_nodes(program_str)

        sketch = SketchSampler().sample_sketch(program_nodes, 4)

        sketch_str = Parser.nodes_to_tokens(sketch)
        
        sketch_tokens = Parser.tokens_to_list(sketch_str)
        
        sketches_list.append((program_info[0], program_info[1], program_info[2], np.array(sketch_tokens)))
    
    with open('data/sketches.pkl', 'wb') as f:
        pickle.dump(sketches_list, f)
    