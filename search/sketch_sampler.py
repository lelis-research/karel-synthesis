from __future__ import annotations
import copy
from typing import Union
import numpy as np

from config import Config
from dsl.base import Node, Program, StatementNode


class SketchSampler:
    
    def __init__(self, seed: int = None, statements_only: bool = False):
        if seed is None:
            self.rng = np.random.RandomState(Config.env_seed)
        else:
            self.rng = np.random.RandomState(seed)
        self.n = Config.datagen_sketch_iterations
        self.statements_only = statements_only
    
    def shrink_node(self, node: Node) -> list[Node]:
        shrunk_nodes = []
        for i, child in enumerate(node.children):
            if child is None or (self.statements_only and not issubclass(type(child), StatementNode)):
                continue
            if child.get_size() == 1:
                # Create a copy of the node without leaf child
                new_node = type(node)()
                new_node.children = copy.deepcopy(node.children)
                new_node.children[i] = None
                shrunk_nodes.append(new_node)
            else:
                # Shrink every leaf of the child node
                shrunk_child = self.shrink_node(child)
                for c in shrunk_child:
                    new_node = type(node)()
                    new_node.children = copy.deepcopy(node.children)
                    new_node.children[i] = c
                    shrunk_nodes.append(new_node)
        return shrunk_nodes
    
    def get_all_sketches(self, program: Program, n: int) -> list[Program]:
        plist = [program]
        all_plist = []
        for _ in range(n):
            new_plist = []
            for p in plist:
                new_plist += self.shrink_node(p)
            plist = new_plist
            all_plist += new_plist
        return all_plist
    
    def sample_sketch(self, program: Program, n: Union[int, None] = None) -> Program:
        if n is None: n = self.n
        all_sketches = self.get_all_sketches(program, n)
        return self.rng.choice(all_sketches)