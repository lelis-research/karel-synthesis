from __future__ import annotations
import copy
from typing import Tuple
import numpy as np
from karel.environment import Environment
from karel.world import World
from dsl.base import *
from dsl.parser import Parser

def sample_sketch(prog: Program, n_cuts: int):
    
    def get_leaf_nodes(node: Node) -> Tuple[list[Node], list[Node], list[int]]:
        leaf_nodes = []
        leaf_nodes_parents = []
        leaf_nodes_indices = []
        for i, child in enumerate(node.children):
            if child is not None:
                if child.get_size() == 1:
                    leaf_nodes.append(child)
                    leaf_nodes_parents.append(node)
                    leaf_nodes_indices.append(i)
                else:
                    result = get_leaf_nodes(child)
                    new_leaf_nodes, new_leaf_nodes_parents, new_leaf_nodes_indices = result
                    leaf_nodes += new_leaf_nodes
                    leaf_nodes_parents += new_leaf_nodes_parents
                    leaf_nodes_indices += new_leaf_nodes_indices
        return leaf_nodes, leaf_nodes_parents, leaf_nodes_indices
    
    print('Original:', Parser.nodes_to_tokens(prog))
    
    cut_prog = copy.deepcopy(prog)
    for n in range(1, n_cuts + 1):        
        prog_leaf_nodes, prog_leaf_nodes_parents, prog_leaf_nodes_indices = get_leaf_nodes(cut_prog)
        index = np.random.randint(len(prog_leaf_nodes))
        prog_leaf_nodes_parents[index].children[prog_leaf_nodes_indices[index]] = None
        
        print(f'After {n} cuts:', Parser.nodes_to_tokens(cut_prog))
        
    return cut_prog

def fill_sketch(prog: Program):
    
    # TODO: Implement top-down search to fill the holes in the program
    
    def get_holes(node: Node) -> Tuple[list[Node], list[Node], list[int]]:
        pass
    
    filled_prog = copy.deepcopy(prog)
    
    return filled_prog

if __name__ == '__main__':
    
    complete_program = Program.new(
        Conjunction.new(
            If.new(RightIsClear(), Conjunction.new(
                TurnRight(), Conjunction.new(
                    Move(), Conjunction.new(
                        PutMarker(), Conjunction.new(
                            TurnLeft(), Conjunction.new(
                                TurnLeft(), Conjunction.new(
                                    Move(), TurnRight()
                                )
                            )
                        )
                    )
                )
            )),
            While.new(FrontIsClear(), Conjunction.new(
                Move(), If.new(RightIsClear(), Conjunction.new(
                    TurnRight(), Conjunction.new(
                        Move(), Conjunction.new(
                            PutMarker(), Conjunction.new(
                                TurnLeft(), Conjunction.new(
                                    TurnLeft(), Conjunction.new(
                                        Move(), TurnRight()
                                    )
                                )
                            )
                        )
                    )
                ))
            ))
        )
    )
    
    cut_prog = sample_sketch(complete_program, 10)

    filled_prog = fill_sketch(cut_prog)

    worlds = [
        '|  |\n' +
        '|  |\n' +
        '|^*|',

        '|  |\n' +
        '| *|\n' +
        '|  |\n' +
        '|  |\n' +
        '| *|\n' +
        '|  |\n' +
        '|  |\n' +
        '| *|\n' +
        '|  |\n' +
        '|^ |',

        '|  |\n' +
        '| *|\n' +
        '| *|\n' +
        '| *|\n' +
        '|  |\n' +
        '|  |\n' +
        '| *|\n' +
        '| *|\n' +
        '| *|\n' +
        '|^ |'
    ]

    for i, w in enumerate(worlds):

        world = World.from_string(w)

        env = Environment(world, program)
        env.run_and_trace(f'output/symbolic_{i}.gif')

        f_world = env.get_world_state()

        print(f_world.to_string())
