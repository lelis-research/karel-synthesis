from __future__ import annotations
from typing import Tuple
import copy
from dsl.base import *
from dsl.production import Production
from tasks.task import Task

class TopDownSearch:

    def get_reward(self, p: Program, num_evaluations: int):
        self._num_evaluations += 1
        mean_reward = 0
        for seed in range(num_evaluations):
            env = self.task.generate_state(seed)
            reward = 0
            for _ in program.run_generator(state):
                terminated, instant_reward = self.task.get_reward(state)
                reward += instant_reward
                if terminated:
                    break
            mean_reward += reward
        mean_reward /= num_evaluations
        return mean_reward
    
    def grow_node(self, node: Node, production: Production):
        for i, child in enumerate(node.children):
            if child is None:
                
                # Apply production rule
            else:
                grown_children = [self.grow_node(child, production)]
                grown_nodes = []
                for c in grown_children:
                    new_children = copy.deepcopy(node.children)
                    new_children[i] = c
                    grown_node = node.__cls__()
                    grown_node.children = new_children
                    grown_nodes.append(grown_node)
                return grown_nodes
    
    def grow_program(self, program: Program, production: Production) -> list[Program]:
        new_plist = []
        # Traverse through whole program tree
        nodes_to_traverse = [new_program]
        nodes_to_grow: list[Tuple[Node, int]] = []
        while len(nodes_to_traverse) > 0:
            current_node = nodes_to_traverse.pop()
            if current_node.is_complete():
                continue
            for i, child in enumerate(node.children):
                if child is None:
                    # Apply production rule
                    nodes_to_grow.append((node, i))
                else:
                    nodes_to_traverse.append(child)
        return new_plist

    def grow(self, plist: list[Program], production: Production):
        print('growing')
        new_plist = []
        for p in plist:
            new_plist += self.grow_program(p, production)
        return new_plist
    
    def synthesize(self, initial_program: Program, production: Production, task: Task, bound: int) -> tuple[Program, int]:
        self._num_evaluations = 0
        
        plist = [obj for obj in production.nodes if obj.__class__ in TerminalNode.__subclasses__()]
        # plist = self.elim_equivalents(plist, data)
        print(f'Iteration 1: {len(plist)} new programs.')
        for p in plist:
            # print(f'Program: {p.to_string()}')
            if self.is_correct(p, data):
                return Program.new(p), self._num_evaluations

        for i in range(2, bound+1):
            new_plist = self.grow(plist, production, i)
            # new_plist = self.elim_equivalents(new_plist, data)
            plist += new_plist
            print(f'Iteration {i}: {len(new_plist)} new programs.')
            print('checking programs')
            for i, p in enumerate(new_plist):
                if self.is_correct(p, data):
                    return Program.new(p), self._num_evaluations
        return None, self._num_evaluations