from __future__ import annotations
from functools import partial
from multiprocessing import Pool
import os
import time
import numpy as np
import torch

from dsl import DSL
from dsl.base import Program
from vae.models.base_vae import BaseVAE
from vae.models.sketch_vae import SketchVAE
from logger.stdout_logger import StdoutLogger
from tasks.task import Task
from config import Config

def evaluate_program(program_tokens: list[int], task_envs: list[Task], dsl: DSL) -> float:
    if program_tokens is None: return -float('inf')
    
    try:
        program = dsl.parse_int_to_node(program_tokens)
    except AssertionError: # In case of invalid program (e.g. does not have an ending token)
        return -float('inf')
        
    rewards = [task_env.evaluate_program(program) for task_env in task_envs]
    
    return np.mean(rewards)


class HierarchicalSearch:
    
    def __init__(self, models: list[BaseVAE], task_cls: type[Task], dsl: DSL):
        self.models: list[BaseVAE] = models
        self.iterations = [Config.hierarchical_level_1_iterations, Config.hierarchical_level_2_iterations]
        self.pop_sizes = [Config.hierarchical_level_1_pop_size, Config.hierarchical_level_2_pop_size]
        self.n_elite = [
            int(Config.search_elitism_rate * self.pop_sizes[0]),
            int(Config.search_elitism_rate * self.pop_sizes[1])
        ]
        self.dsl = dsl
        self.sketch_dsl = dsl.extend_dsl()
        self.device = self.models[0].device

        self.number_executions = Config.search_number_executions
        self.number_iterations = Config.search_number_iterations
        self.sigma = Config.search_sigma
        self.task_envs = [task_cls(i) for i in range(self.number_executions)]
        output_dir = os.path.join('output', Config.experiment_name, 'latent_search')
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, f'seed_{Config.model_seed}.csv')
        self.trace_file = os.path.join(output_dir, f'seed_{Config.model_seed}.gif')
        self.restart_timeout = Config.search_restart_timeout


    def evaluate_program(self, program_tokens: list[int]) -> float:
        if program_tokens is None: return -float('inf')

        try:
            program = self.dsl.parse_int_to_node(program_tokens)
        except AssertionError: # In case of invalid program (e.g. does not have an ending token)
            return -float('inf')
        
        sum_reward = 0.
        for task_env in self.task_envs:
            reward = task_env.evaluate_program(program)
            if reward < self.best_reward:
                return -float('inf')
            sum_reward += reward
        
        return sum_reward / len(self.task_envs)


    def get_program(self, sketch_tokens: list[int], holes_tokens: list[list[int]]) -> list[int]:
        # split sketch_tokens into multiple lists based on <HOLE> token
        list_sketch = [[]]
        for token in sketch_tokens:
            if token == self.sketch_dsl.t2i['<HOLE>']:
                list_sketch.append([])
            else:
                list_sketch[-1].append(token)
        assert len(list_sketch) == len(holes_tokens) + 1
        prog_tokens = list_sketch[0]
        for i in range(len(holes_tokens)):
            assert holes_tokens[i][0] == self.dsl.t2i['DEF']
            assert holes_tokens[i][1] == self.dsl.t2i['run']
            assert holes_tokens[i][2] == self.dsl.t2i['m(']
            assert holes_tokens[i][-1] == self.dsl.t2i['m)']
            prog_tokens += holes_tokens[i][3:-1] + list_sketch[i+1]
        return prog_tokens


    def recursive_search(self, level: int, current_program: list[int]):
        n_holes = current_program.count(self.sketch_dsl.t2i['<HOLE>'])
    
        population = torch.stack([
            torch.stack([
                torch.randn(self.models[level].hidden_size, device=self.device)
                for _ in range(n_holes)
            ])
            for _ in range(self.pop_sizes[level])
        ])
        best_program = None
        best_reward = -float('inf')
        prev_mean_elite_reward = -float('inf')
        
        for iteration in range(1, self.iterations[level] + 1):
            StdoutLogger.log('Hierarchical Search', f'Level {level} Iteration {iteration}.')
            population_holes_tokens = [
                [self.models[level].decode_vector(hole.unsqueeze(0))[0] for hole in p]
                for p in population
            ]
            
            filled_programs = []
            for holes_tokens in population_holes_tokens:
                try:
                    filled_program = self.get_program(current_program, holes_tokens)
                    filled_programs.append(filled_program)
                except AssertionError:
                    filled_programs.append(None)

            # If the program is still incomplete, we call this function recursively
            if type(self.models[level]) == SketchVAE:
                rewards = []
                completed_programs = []
                for filled_program in filled_programs:
                    completed_program, reward = self.recursive_search(level + 1, filled_program)
                    rewards.append(reward)
                    completed_programs.append(completed_program)
                    if self.converged:
                        break
            # In case it is the last level of hierarchical search
            else:
                completed_programs = filled_programs
                if Config.multiprocessing_active:
                    with Pool() as pool:
                        fn = partial(evaluate_program, task_envs=self.task_envs, dsl=self.dsl)
                        rewards = pool.map(fn, completed_programs)
                else:
                    rewards = [evaluate_program(p, self.task_envs, self.dsl) for p in completed_programs]
                self.num_eval += len(rewards)
            
            rewards = torch.tensor(rewards)
            
            if torch.max(rewards) > best_reward:
                best_reward = torch.max(rewards)
                best_program = completed_programs[torch.argmax(rewards)]
                
            if best_reward > self.best_reward:
                self.best_reward = best_reward
                self.best_program = self.dsl.parse_int_to_str(best_program)
                StdoutLogger.log('Hierarchical Search', f'New best program: {self.best_program}')
                StdoutLogger.log('Hierarchical Search', f'New best reward: {self.best_reward}')
                StdoutLogger.log('Hierarchical Search', f'Number of evaluations: {self.num_eval}')
                with open(self.output_file, mode='a') as f:
                    t = time.time() - self.start_time
                    f.write(f'{t},{self.num_eval},{self.best_reward},{self.best_program}\n')
            
            if self.best_reward >= 1.0:
                self.converged = True
                break
            # TODO: break out of recursion when converged

            best_indices = torch.topk(rewards, self.n_elite[level]).indices
            elite_population = population[best_indices]
            mean_elite_reward = torch.mean(rewards[best_indices])
            
            if mean_elite_reward.cpu().numpy() == prev_mean_elite_reward:
                counter_for_restart += 1
            else:
                counter_for_restart = 0
            
            StdoutLogger.log('Hierarchical Search', f'Mean Elite Reward: {mean_elite_reward}')
            StdoutLogger.log('Hierarchical Search', f'Num eval so far: {self.num_eval}')
            
            if counter_for_restart >= self.restart_timeout and self.restart_timeout > 0:
                StdoutLogger.log('Hierarchical Search', f'Restarted population for level {level} search.')
                population = torch.stack([
                    torch.stack([
                        torch.randn(self.models[level].hidden_size, device=self.device)
                        for _ in range(n_holes)
                    ])
                    for _ in range(self.pop_sizes[level])
                ])
                counter_for_restart = 0
            else:
                new_indices = torch.ones(elite_population.size(0), device=self.device).multinomial(
                    self.pop_sizes[level], replacement=True)
                new_population = []
                for index in new_indices:
                    sample = elite_population[index]
                    new_population.append(
                        sample + self.sigma * torch.randn_like(sample, device=self.device)
                    )
                population = torch.stack(new_population)
            prev_mean_elite_reward = mean_elite_reward.cpu().numpy()
        return best_program, best_reward

    
    def search(self) -> tuple[str, bool, int]:
        self.converged = False
        self.num_eval = 0
        self.best_reward = float('-inf')
        self.best_program = None
        self.start_time = time.time()
        with open(self.output_file, mode='w') as f:
            f.write('time,num_evaluations,best_reward,best_program\n')
        
        # for i in range(1, self.number_iterations + 1):
        program = self.sketch_dsl.parse_str_to_int('DEF run m( <HOLE> m)')
        _, _ = self.recursive_search(0, program)
        # best_program_str = self.dsl.parse_int_to_str(best_program)
        
        # if self.converged: break
        
        # StdoutLogger.log('Hierarchical Search',f'Iteration {i}:')
        # StdoutLogger.log('Hierarchical Search',f'Best reward: {best_reward}')
        # StdoutLogger.log('Hierarchical Search',f'Best program: {best_program_str}')
        # StdoutLogger.log('Hierarchical Search',f'Number of evaluations: {self.num_eval}')

        best_program_nodes = self.dsl.parse_str_to_node(self.best_program)
        self.task_envs[0].trace_program(best_program_nodes, self.trace_file, 1000)

        return self.best_program, self.converged, self.num_eval
