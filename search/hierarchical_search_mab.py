from __future__ import annotations
from functools import partial
import math
from multiprocessing import Pool
import os
import time
import numpy as np
import torch

from dsl import DSL
from vae.models.base_vae import BaseVAE
from vae.models.sketch_vae import SketchVAE
from logger.stdout_logger import StdoutLogger
from tasks.task import Task
from config import Config


def evaluate_program(program_tokens: list[int], task_envs: list[Task], dsl: DSL) -> float:
    try:
        program = dsl.parse_int_to_node(program_tokens)
    except AssertionError: # In case of invalid program (e.g. does not have an ending token)
        return -float('inf')
    
    sum_reward = 0.
    for task_env in task_envs:
        sum_reward += task_env.evaluate_program(program)
    
    return sum_reward / len(task_envs)


class HierarchicalSearchMAB:
    
    def __init__(self, models: list[BaseVAE], task_cls: type[Task], dsl: DSL, seed: int = None):
        self.models: list[BaseVAE] = models
        self.search_iterations = Config.hierarchical_mab_search_iterations
        self.sample_iterations = Config.hierarchical_mab_sample_iterations
        self.pop_sizes = [Config.hierarchical_level_1_pop_size, Config.hierarchical_level_2_pop_size]
        self.elitism_rate = Config.search_elitism_rate
        self.n_elite = [
            int(Config.search_elitism_rate * self.pop_sizes[0]),
            int(Config.search_elitism_rate * self.pop_sizes[1])
        ]
        self.dsl = dsl.extend_dsl()
        self.device = self.models[0].device
        self.batch_size = Config.hierarchical_mab_batch_size
        self.n_holes_constraint = [2, 0]

        self.number_executions = Config.search_number_executions
        self.sigma = Config.search_sigma
        self.epsilon = Config.hierarchical_mab_epsilon
        self.task_envs = [task_cls(i) for i in range(self.number_executions)]
        output_dir = os.path.join('output', Config.experiment_name, 'latent_search')
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, f'seed_{Config.model_seed}.csv')
        self.trace_file = os.path.join(output_dir, f'seed_{Config.model_seed}.gif')
        self.restart_timeout = Config.search_restart_timeout
        if seed is None:
            self.rng = np.random.default_rng(Config.model_seed)
        else:
            self.rng = np.random.default_rng(seed)


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
        for i in range(len(holes_tokens)):
            prog_tokens += holes_tokens[i][3:-1] + list_sketch[i+1]
        return prog_tokens
    
    
    def init_random_population(self):
        latent_population = []
        decoded_population = []
        for level in range(len(self.models)):
            latent_level = []
            decoded_level = []
            seen_programs = set()
            while len(latent_level) < self.pop_sizes[level]:
                latent = torch.randn(self.models[level].hidden_size, device=self.device)
                decoded = self.models[level].decode_vector(latent.unsqueeze(0))[0]
                n_holes = decoded.count(self.dsl.t2i['<HOLE>'])
                if n_holes != self.n_holes_constraint[level]: continue
                program_str = self.dsl.parse_int_to_str(decoded)
                if program_str in seen_programs: continue
                if program_str.startswith('DEF run m(') and program_str.endswith('m)'):
                    seen_programs.add(program_str)
                    latent_level.append(latent)
                    decoded_level.append(decoded)
            latent_population.append(torch.stack(latent_level))
            decoded_population.append(decoded_level)
        return latent_population, decoded_population
    
    
    def sample_population_from_elite(self, elite_population):
        latent_population = []
        decoded_population = []
        for level in range(len(self.models)):
            new_latent_level = []
            new_decoded_level = []
            seen_programs = set()
            while len(new_latent_level) < self.pop_sizes[level]:
                sample = elite_population[level][self.rng.choice(len(elite_population[level]), 1)[0]]
                new_latent = sample + self.sigma * torch.randn_like(sample, device=self.device)
                new_decoded = self.models[level].decode_vector(new_latent.unsqueeze(0))[0]
                n_holes = new_decoded.count(self.dsl.t2i['<HOLE>'])
                if n_holes != self.n_holes_constraint[level]: continue
                program_str = self.dsl.parse_int_to_str(new_decoded)
                if program_str in seen_programs: continue
                if program_str.startswith('DEF run m(') and program_str.endswith('m)'):
                    seen_programs.add(program_str)
                    new_latent_level.append(new_latent)
                    new_decoded_level.append(new_decoded)
            latent_population.append(torch.stack(new_latent_level))
            decoded_population.append(new_decoded_level)
        return latent_population, decoded_population
    
    
    def argmax(self, q_values):
        """
        Takes in a list of q_values and returns the index of the item 
        with the highest value. Breaks ties randomly.
        returns: int - the index of the highest value in q_values
        """
        top_value = float("-inf")
        ties = []
        
        for i in range(len(q_values)):
            # if a value in q_values is greater than the highest value update top and reset ties to zero
            # if a value is equal to top value add the index to ties
            # return a random selection from ties.
            if q_values[i] > top_value:
                ties = [i]
                top_value = q_values[i]
            elif q_values[i] == top_value:
                ties.append(i)
        return self.rng.choice(ties)
    
    
    def step(self, decoded_population: list[list[list[int]]]):
        global_q_values = [[] for _ in self.models]
        global_arms = [[] for _ in self.models]
        global_count_calls = [[] for _ in self.models]
        local_q_values = [[0.0 for _ in range(len(pop))] for pop in decoded_population]
        local_count_calls = [[0 for _ in range(len(pop))] for pop in decoded_population]
        iteration_best_reward = float('-inf')
        
        for sample_iteration in range(1, self.sample_iterations + 1):
            
            # Sample programs from decoded population
            programs, local_indices, global_indices = [], [], []
            new_global_arms = [[] for _ in self.models]
            for _ in range(self.batch_size):
                # Initial program
                program = [self.dsl.t2i['DEF'], self.dsl.t2i['run'], self.dsl.t2i['m('],
                           self.dsl.t2i['<HOLE>'], self.dsl.t2i['m)']]
                this_local_indices, this_global_indices = [], []
                for level in range(len(self.models)):
                    n_holes = program.count(self.dsl.t2i['<HOLE>'])
                    holes, holes_indices = [], []
                    if n_holes > 1:
                        # CMAB Naive Sampling
                        if self.rng.random() < self.epsilon or len(global_arms[level]) == 0: # Outer epsilon-greedy
                            # Exploration of new global arm using local selection
                            for _ in range(n_holes):
                                if self.rng.random() < self.epsilon: # Local epsilon-greedy
                                    hole_index = self.rng.choice(list(range(len(decoded_population[level]))))
                                else:
                                    hole_index = self.argmax(local_q_values[level])
                                holes_indices.append(hole_index)
                                holes.append(decoded_population[level][hole_index])
                            # Check if global arm does not already exist: this might be not necessary for large enough population
                            if holes_indices not in global_arms[level]:
                                if holes_indices not in new_global_arms[level]:
                                    # Add new global arm
                                    new_global_arms[level].append(holes_indices)
                                    global_index = len(global_arms[level]) + len(new_global_arms[level]) - 1
                                else:
                                    # Index of existing new global arm
                                    global_index = new_global_arms[level].index(holes_indices) + len(global_arms[level])
                            else:
                                # Index of existing global arm
                                
                                global_index = global_arms[level].index(holes_indices)
                        else:
                            # Select from seen global arms
                            if self.rng.random() < self.epsilon: # Global epsilon-greedy
                                global_index = self.rng.choice(list(range(len(global_arms[level]))))
                            else:
                                global_index = self.argmax(global_q_values[level])
                            holes_indices = global_arms[level][global_index]
                        this_global_indices.append(global_index)
                    else:
                        # Single MAB Epsilon-Greedy Selection
                        if self.rng.random() < self.epsilon:
                            hole_index = self.rng.choice(list(range(len(decoded_population[level]))))
                        else:
                            hole_index = self.argmax(local_q_values[level])
                        holes_indices.append(hole_index)
                        this_global_indices.append(None)
    
                    holes = [decoded_population[level][hole_index] for hole_index in holes_indices]
                    program = self.get_program(program, holes)
                    this_local_indices.append(holes_indices)
                programs.append(program)
                local_indices.append(this_local_indices)
                global_indices.append(this_global_indices)
            
            for level in range(len(self.models)):
                global_arms[level].extend(new_global_arms[level])
                global_q_values[level].extend([0.0 for _ in new_global_arms[level]])
                global_count_calls[level].extend([0 for _ in new_global_arms[level]])
            
            # Evaluate the programs
            if Config.multiprocessing_active:
                fn = partial(evaluate_program, task_envs=self.task_envs, dsl=self.dsl)
                rewards = self.pool.map(fn, programs)
            else:
                rewards = [evaluate_program(program, self.task_envs, self.dsl) for program in programs]
            self.num_eval += len(rewards)
            StdoutLogger.log('Hierarchical Search', f'Sample iteration {sample_iteration} best reward: {max(rewards)}')
            
            if max(rewards) > iteration_best_reward:
                iteration_best_reward = max(rewards)
            
            # Update best program
            if max(rewards) > self.best_reward:
                self.best_reward = max(rewards)
                self.best_program = self.dsl.parse_int_to_str(programs[np.argmax(rewards)])
                StdoutLogger.log('Hierarchical Search', f'New best program: {self.best_program}')
                StdoutLogger.log('Hierarchical Search', f'New best reward: {self.best_reward}')
                StdoutLogger.log('Hierarchical Search', f'Number of evaluations: {self.num_eval}')
                with open(self.output_file, mode='a') as f:
                    t = time.time() - self.start_time
                    f.write(f'{t},{self.num_eval},{self.best_reward},{self.best_program}\n')
            
            if self.best_reward >= 1.0:
                self.converged = True
                break
            
            # Update Local Q-values
            for reward, indices in zip(rewards, local_indices):
                for level in range(len(self.models)):
                    for hole_index in indices[level]:
                        local_count_calls[level][hole_index] += 1
                        local_q_values[level][hole_index] += (reward - local_q_values[level][hole_index]) / local_count_calls[level][hole_index]

            # Update Global Q-values
            for reward, indices in zip(rewards, global_indices):
                for level in range(len(self.models)):
                    if indices[level] is None: continue
                    global_index = indices[level]
                    global_count_calls[level][global_index] += 1
                    global_q_values[level][global_index] += (reward - global_q_values[level][global_index]) / global_count_calls[level][global_index]

        return local_q_values, iteration_best_reward
    
    
    def search(self) -> tuple[str, bool, int]:
        self.converged = False
        self.num_eval = 0
        self.best_reward = float('-inf')
        self.best_program = None
        self.start_time = time.time()
        if Config.multiprocessing_active: self.pool = Pool()

        prev_best_reward = float('-inf')
        restart_counter = 0
        with open(self.output_file, mode='w') as f:
            f.write('time,num_evaluations,best_reward,best_program\n')

        latent_population, decoded_population = self.init_random_population()
        
        for search_iteration in range(1, self.search_iterations + 1):
        
            StdoutLogger.log('Hierarchical Search', f'Search iteration {search_iteration}.')
            StdoutLogger.log('Hierarchical Search', f'Estimating Q-values.')
            current_num_eval = self.num_eval
        
            # Estimate Q-values by executing sampled programs
            q_values, iteration_best_reward = self.step(decoded_population)
            
            if self.converged: break
            
            StdoutLogger.log('Hierarchical Search', f'Number of evaluations in iteration {search_iteration}: {self.num_eval - current_num_eval}')
            StdoutLogger.log('Hierarchical Search', f'Sampling next population.')
            
            if iteration_best_reward == prev_best_reward:
                restart_counter += 1
            else:
                restart_counter = 0
            
            if restart_counter >= self.restart_timeout and self.restart_timeout > 0:
                StdoutLogger.log('Hierarchical Search', f'Restarting search.')
                # Sample new population
                latent_population, decoded_population = self.init_random_population()
                restart_counter = 0
            else:
                # Get elite population based on Q-values
                elite_population = []
                for level in range(len(self.models)):
                    level_elite_population = []
                    n_elite = math.ceil(len(latent_population[level]) * self.elitism_rate)
                    elite_indices = np.argpartition(q_values[level], -n_elite)[-n_elite:]
                    for index in elite_indices:
                        level_elite_population.append(latent_population[level][index])
                    elite_population.append(torch.stack(level_elite_population))
                
                # Sample new population using CEM
                latent_population, decoded_population = self.sample_population_from_elite(elite_population)
                        
            prev_best_reward = iteration_best_reward

        best_program_nodes = self.dsl.parse_str_to_node(self.best_program)
        self.task_envs[0].trace_program(best_program_nodes, self.trace_file, 1000)
        if Config.multiprocessing_active: self.pool.close()
        
        if not self.converged:
            # Report last best program
            with open(self.output_file, mode='a') as f:
                t = time.time() - self.start_time
                f.write(f'{t},{self.num_eval},{self.best_reward},{self.best_program}\n')
        
        return self.best_program, self.converged, self.num_eval
