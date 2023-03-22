from __future__ import annotations
from dsl.base import *
from dsl import DSL
from karel.data import Data
from karel.environment import Environment
import itertools

# Warning: outdated class (uses programming by example instead of Task reward)
class BottomUpSearch:

    # elim_equivalents is not used in this class as I could not adapt it for Karel
    def elim_equivalents(self, plist: list[Program], data: Data):
        unique_plist = []
        for p in plist:
            for inp in data.inputs:
                env = Environment(inp, p)
                actions = env.run_agent()
                if actions not in self._outputs:
                    self._outputs.append(env)
                    unique_plist.append(p)
        return unique_plist

    def is_correct(self, p: Program, data: Data):
        self._num_evaluations += 1
        for inp, out in zip(data.inputs, data.targets):
            try:
                env = Environment(inp, p)
            except:
                return False # In case p cannot be executed as a program
            env.run_agent()
            if not (env.get_world_state() == out):
                return False
        return True

    def grow_node(self, node: Node, plist: list[Node], new_plist: list[Node], size: int):
        types = node.get_children_types()
        possible_children = []
        for child_type in types:
            possible_children.append([p for p in plist if p.__class__ in child_type.__subclasses__()])
        combinations = itertools.product(*possible_children)
        for comb in combinations:
            new_node = node.__class__()
            for child in comb:
                new_node.add_child(child)
            if new_node.get_size() == size:
                new_plist.append(new_node)

    def grow(self, plist: list[Node], production: DSL, size: int):
        new_plist = []
        print('growing')
        for op in [obj for obj in production.nodes if obj.__class__ not in TerminalNode.__subclasses__()]:
            self.grow_node(op, plist, new_plist, size)
        return new_plist
    
    def synthesize(self, data: Data, production: DSL, bound) -> tuple[Node, int]:
        self._num_evaluations = 0
        self._outputs = []

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