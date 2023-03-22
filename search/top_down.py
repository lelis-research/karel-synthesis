from __future__ import annotations
import copy
from multiprocessing import Pool
from config import Config
from dsl.base import Node, Program
from dsl import DSL, Production
from logger.stdout_logger import StdoutLogger
from tasks import Task

class TopDownSearch:

    def get_reward(self, p: Program) -> float:
        mean_reward = 0
        for task_env in self.task_envs:
            task_env.reset_state()
            state = task_env.get_state()
            reward = 0
            for _ in p.run_generator(state):
                terminated, instant_reward = task_env.get_reward(state)
                reward += instant_reward
                if terminated:
                    break
            if reward < self.best_reward:
                return -float('inf')
            mean_reward += reward
        mean_reward /= len(self.task_envs)
        return mean_reward
    
    def get_number_holes(self, node: Node) -> int:
        holes = 0
        for child in node.children:
            if child is None:
                holes += 1
            else:
                holes += self.get_number_holes(child)
        return holes
    
    def grow_node(self, node: Node, grow_bound: int) -> list[Node]:
        n_holes = self.get_number_holes(node)
        if n_holes == 0:
            return []

        grown_children = []
        prod_rule = Production.get_production_rules(type(node))
        for i, child in enumerate(node.children):
            if child is None:
                # Replace child with possible production rules
                child_type = type(node).get_children_types()[i]
                child_list = [n for n in self.dsl.nodes 
                              if type(n) in prod_rule[i]
                              and n.get_size() + len(n.children) <= grow_bound - n_holes + 1]
                grown_children.append(child_list)
            else:
                grown_children.append(self.grow_node(child, grow_bound))
                
        grown_nodes = []
        for i, grown_child in enumerate(grown_children):
            for c in grown_child:
                grown_node = type(node)()
                grown_node.children = copy.deepcopy(node.children)
                grown_node.children[i] = c
                grown_nodes.append(grown_node)
        return grown_nodes
    
    def synthesize(self, initial_program: Program, dsl: DSL, task_cls: type[Task],
                   grow_bound: int = 5) -> tuple[Program, int, bool]:
        self.dsl = dsl
        self.task_envs = [task_cls(i) for i in range(Config.search_number_executions)]
        
        self.best_reward = -float('inf')
        self.best_program = None
        self.converged = False
        
        StdoutLogger.log('Top Down Searcher', f'Initializing top down search with grow bound {grow_bound}.')
        num_evaluations = 0
        plist = [initial_program]

        for i in range(grow_bound):
            # Grow programs once
            new_plist = []
            for p in plist:
                new_plist += self.grow_node(p, grow_bound - i)
            plist = new_plist
            # Evaluate programs
            complete_programs = [p for p in plist if p.is_complete()]
            if Config.search_use_multiprocessing:
                with Pool() as pool:
                    rewards = pool.map(self.get_reward, complete_programs)
            else:
                rewards = [self.get_reward(p) for p in complete_programs]
            num_evaluations += len(complete_programs)
            for p, r in zip(complete_programs, rewards):
                if r > self.best_reward:
                    self.best_reward = r
                    self.best_program = p
            StdoutLogger.log('Top Down Searcher', f'Iteration {i+1}: {len(plist)} new programs.')
            StdoutLogger.log('Top Down Searcher', f'Best reward: {self.best_reward}.')
            StdoutLogger.log('Top Down Searcher', f'Num evaluations: {num_evaluations}.')
            if self.best_reward == 1:
                self.converged = True
                break

        StdoutLogger.log('Top Down Searcher', f'Search converged after {num_evaluations} evaluations.')
        return self.best_program, num_evaluations, self.converged