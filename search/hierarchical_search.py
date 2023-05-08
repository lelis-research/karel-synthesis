from __future__ import annotations
from functools import partial
import math
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
        self.elitism_rate = Config.search_elitism_rate
        self.dsl = dsl.extend_dsl()
        self.device = self.models[0].device

        self.sigma = Config.search_sigma
        self.task_envs = [task_cls(i) for i in range(Config.search_number_executions)]
        output_dir = os.path.join('output', Config.experiment_name, 'latent_search')
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, f'seed_{Config.model_seed}.csv')
        self.trace_file = os.path.join(output_dir, f'seed_{Config.model_seed}.gif')
        self.restart_timeout = Config.search_restart_timeout
    
    
    def init_random_population(self, level: int, n_holes: int):
        population = torch.stack([
            torch.stack([
                torch.randn(self.models[level].hidden_size, device=self.device)
                for _ in range(n_holes)
            ])
            for _ in range(self.pop_sizes[level])
        ])
        
        decoded_population = [[] for _ in range(n_holes) for _ in range(self.pop_sizes[level])]
        for hole in range(n_holes):
            decoded_population_list = self.models[level].decode_vector(population[:, hole])
            for dp, dpl in zip(decoded_population, decoded_population_list):
                dp.append(dpl)
        
        return population, decoded_population
    
    
    def remove_invalid(self, latent_population: list[torch.Tensor],
                             decoded_population: list[list[list[int]]]):
        new_latent_population, new_decoded_population = [], []
        # seen_programs = set()
        for latent_level, decoded_level in zip(latent_population, decoded_population):
            latent_level_list, decoded_level_list = [], []
            for latent, decoded in zip(latent_level, decoded_level):
                program_str = self.dsl.parse_int_to_str(decoded)
                # if program_str in seen_programs: continue
                if program_str.startswith('DEF run m(') and program_str.endswith('m)'):
                    # seen_programs.add(program_str)
                    latent_level_list.append(latent)
                    decoded_level_list.append(decoded)
            if len(decoded_level_list) == len(decoded_level): # No invalid programs
                new_latent_population.append(torch.stack(latent_level_list))
                new_decoded_population.append(decoded_level_list)
        if len(new_decoded_population) == 0:
            return None, None
        return torch.stack(new_latent_population), new_decoded_population


    def get_program(self, sketch_tokens: list[int], holes_tokens: list[list[int]]) -> list[int]:
        # split sketch_tokens into multiple lists based on <HOLE> token
        list_sketch = [[]]
        for token in sketch_tokens:
            if token == self.dsl.t2i['<HOLE>']:
                list_sketch.append([])
            else:
                list_sketch[-1].append(token)
        assert len(list_sketch) == len(holes_tokens) + 1
        prog_tokens = list_sketch[0]
        # Removing DEF run m( m) tokens from holes_tokens
        for i in range(len(holes_tokens)):
            prog_tokens += holes_tokens[i][3:-1] + list_sketch[i+1]
        return prog_tokens


    def recursive_search(self, level: int, current_sketch: list[int]):
        n_holes = current_sketch.count(self.dsl.t2i['<HOLE>'])
    
        population, decoded_population = self.init_random_population(level, n_holes)
        population, decoded_population = self.remove_invalid(population, decoded_population)
        best_program = None
        best_reward = -float('inf')
        prev_mean_elite_reward = -float('inf')
        
        for iteration in range(1, self.iterations[level] + 1):
            StdoutLogger.log('Hierarchical Search', f'Level {level} Iteration {iteration}.')
            
            filled_programs = []
            for decoded_program in decoded_population:
                try:
                    filled_program = self.get_program(current_sketch, decoded_program)
                    filled_programs.append(filled_program)
                except AssertionError:
                    filled_programs.append(None)

            complete_programs, incomplete_programs = [], []
            for filled_program in filled_programs:
                filled_prog_n_holes = filled_program.count(self.dsl.t2i['<HOLE>'])
                if filled_prog_n_holes == 0:
                    complete_programs.append(filled_program)
                else:
                    incomplete_programs.append(filled_program)                    
            
            complete_programs_rewards, incomplete_programs_rewards = [], []
            if len(complete_programs) > 0:
                if Config.multiprocessing_active:
                    fn = partial(evaluate_program, task_envs=self.task_envs, dsl=self.dsl)
                    complete_programs_rewards = self.pool.map(fn, complete_programs)
                else:
                    complete_programs_rewards = [evaluate_program(p, self.task_envs, self.dsl) for p in complete_programs]
                self.num_eval += len(complete_programs_rewards)
            
            recursively_completed_programs = []
            if len(incomplete_programs) > 0:
                for incomplete_program in incomplete_programs:
                    completed_program, reward = self.recursive_search(level + 1, incomplete_program)
                    recursively_completed_programs.append(completed_program)
                    incomplete_programs_rewards.append(reward)
                    if self.converged:
                        break

            programs = complete_programs + recursively_completed_programs
            rewards = torch.tensor(complete_programs_rewards + incomplete_programs_rewards, device=self.device)
            
            if torch.max(rewards) > best_reward:
                best_reward = torch.max(rewards)
                best_program = programs[torch.argmax(rewards)]
                
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
                return best_program, best_reward

            n_elite = math.ceil(len(population) * self.elitism_rate)
            best_indices = torch.topk(rewards, n_elite).indices
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
                population, decoded_population = self.init_random_population(level, n_holes)
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
                decoded_population = [[] for _ in range(n_holes) for _ in range(self.pop_sizes[level])]
                for hole in range(n_holes):
                    decoded_population_list = self.models[level].decode_vector(population[:, hole])
                    for dp, dpl in zip(decoded_population, decoded_population_list):
                        dp.append(dpl)
            
            population, decoded_population = self.remove_invalid(population, decoded_population)
            
            if population is None:
                StdoutLogger.log('Hierarchical Search', f'No valid programs found, restarting level {level} search.')
                population, decoded_population = self.init_random_population(level, n_holes)
                population, decoded_population = self.remove_invalid(population, decoded_population)
            
            prev_mean_elite_reward = mean_elite_reward.cpu().numpy()
        return best_program, best_reward

    
    def search(self) -> tuple[str, bool, int]:
        self.converged = False
        self.num_eval = 0
        self.best_reward = float('-inf')
        self.best_program = None
        self.start_time = time.time()
        
        if Config.multiprocessing_active: self.pool = Pool()
        
        with open(self.output_file, mode='w') as f:
            f.write('time,num_evaluations,best_reward,best_program\n')
        
        program = self.dsl.parse_str_to_int('DEF run m( <HOLE> m)')
        _, _ = self.recursive_search(0, program)

        best_program_nodes = self.dsl.parse_str_to_node(self.best_program)
        self.task_envs[0].trace_program(best_program_nodes, self.trace_file, 1000)
        
        if Config.multiprocessing_active: self.pool.close()

        return self.best_program, self.converged, self.num_eval
