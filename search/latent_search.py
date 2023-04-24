from __future__ import annotations
from functools import partial
from multiprocessing import Pool
import os
import time
import torch

from dsl import DSL
from dsl.base import Program
from search.top_down import TopDownSearch
from vae.models.base_vae import BaseVAE
from logger.stdout_logger import StdoutLogger
from tasks.task import Task
from config import Config


def execute_program(program_str: str, task_envs: list[Task], task_cls: type[Task],
                    dsl: DSL) -> tuple[Program, int, float]:
    try:
        program = dsl.parse_str_to_node(program_str)
    except AssertionError: # In case of invalid program (e.g. does not have an ending token)
        return Program(), 0, -float('inf')
    # If program is a sketch
    if not program.is_complete():
        # Let TDS complete and evaluate programs
        tds = TopDownSearch()
        tds_result = tds.synthesize(program, dsl, task_cls, Config.datagen_sketch_iterations)
        program, num_evaluations, mean_reward = tds_result
        if program is None: # Failsafe for TDS result
            return program, num_evaluations, -float('inf')
        if not program.is_complete(): # TDS failed to complete program
            return program, num_evaluations, -float('inf')
    # If program is a complete program
    else:
        # Evaluate single program
        num_evaluations = 0
        mean_reward = 0.
        for task_env in task_envs:
            mean_reward += task_env.evaluate_program(program)
            num_evaluations += 1
        mean_reward /= len(task_envs)
    return program, num_evaluations, mean_reward


class LatentSearch:
    """Implements the CEM method from LEAPS paper.
    """
    def __init__(self, model: BaseVAE, task_cls: type[Task], dsl: DSL):
        self.model = model
        self.dsl = dsl
        self.device = self.model.device
        self.population_size = Config.search_population_size
        self.elitism_rate = Config.search_elitism_rate
        self.n_elite = int(Config.search_elitism_rate * self.population_size)
        self.number_executions = Config.search_number_executions
        self.number_iterations = Config.search_number_iterations
        self.sigma = Config.search_sigma
        self.model_hidden_size = Config.model_hidden_size
        self.task_cls = task_cls
        self.task_envs = [self.task_cls(i) for i in range(self.number_executions)]
        self.program_filler = TopDownSearch()
        self.filler_iterations = Config.datagen_sketch_iterations
        output_dir = os.path.join('output', Config.experiment_name, 'latent_search')
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, f'seed_{Config.model_seed}.csv')
        self.restart_timeout = Config.search_restart_timeout


    def init_population(self) -> torch.Tensor:
        """Initializes the CEM population from a normal distribution.

        Returns:
            torch.Tensor: Initial population as a tensor.
        """
        return torch.stack([
            torch.randn(self.model_hidden_size, device=self.device) for _ in range(self.population_size)
        ])
        
        
    def execute_population(self, population: torch.Tensor) -> tuple[list[str], torch.Tensor, int]:
        """Runs the given population in the environment and returns a list of mean rewards, after
        `Config.search_number_executions` executions.

        Args:
            population (torch.Tensor): Current population as a tensor.

        Returns:
            tuple[list[str], int, torch.Tensor]: List of programs as strings, list of mean rewards
            as tensor and number of evaluations as int.
        """
        programs_tokens = self.model.decode_vector(population)
        programs_str = [self.dsl.parse_int_to_str(prog_tokens) for prog_tokens in programs_tokens]
        
        if Config.multiprocessing_active:
            with Pool() as pool:
                fn = partial(execute_program, task_envs=self.task_envs, task_cls=self.task_cls, dsl=self.dsl)
                results = pool.map(fn, programs_str)
        else:
            results = [execute_program(p, self.task_envs, self.task_cls, self.dsl) for p in programs_str]
        
        results_programs_str = []
        rewards = []
        num_evaluations = 0
        for p, num_eval, r in results:
            results_programs_str.append(self.dsl.parse_node_to_str(p))
            rewards.append(r)
            num_evaluations += num_eval
        
        return results_programs_str, num_evaluations, torch.tensor(rewards, device=self.device)

    
    def search(self) -> tuple[str, bool, int]:
        """Main search method. Searches for a program using the specified DSL that yields the
        highest reward at the specified task.

        Returns:
            tuple[str, bool]: Best program in string format and a boolean value indicating
            if the search has converged.
        """
        population = self.init_population()
        converged = False
        num_evaluations = 0
        counter_for_restart = 0
        prev_mean_elite_reward = -float('inf')
        start_time = time.time()
        with open(self.output_file, mode='w') as f:
            f.write('iteration,time,mean_elite_reward,num_evaluations,best_reward,best_program\n')

        for iteration in range(1, self.number_iterations + 1):
            programs, num_eval, rewards = self.execute_population(population)
            best_indices = torch.topk(rewards, self.n_elite).indices
            elite_population = population[best_indices]
            mean_elite_reward = torch.mean(rewards[best_indices])
            best_program = programs[torch.argmax(rewards)]
            best_reward = torch.max(rewards)
            num_evaluations += num_eval
            with open(self.output_file, mode='a') as f:
                t = time.time() - start_time
                f.write(f'{iteration},{t},{mean_elite_reward},{num_eval},{best_reward},{best_program}\n')

            StdoutLogger.log('Latent Search',f'Iteration {iteration}.')
            StdoutLogger.log('Latent Search',f'Mean elite reward: {mean_elite_reward}')
            StdoutLogger.log('Latent Search',f'Number of evaluations in this iteration: {num_eval}')
            StdoutLogger.log('Latent Search',f'Best reward: {best_reward}')
            StdoutLogger.log('Latent Search',f'Best program: {best_program}')
            if mean_elite_reward.cpu().numpy() >= 1.0:
                converged = True
                break
            if mean_elite_reward.cpu().numpy() == prev_mean_elite_reward:
                counter_for_restart += 1
            else:
                counter_for_restart = 0
            if counter_for_restart >= self.restart_timeout and self.restart_timeout > 0:
                population = self.init_population()
                counter_for_restart = 0
                StdoutLogger.log('Latent Search','Restarted population.')
            else:
                new_indices = torch.ones(elite_population.size(0), device=self.device).multinomial(
                    self.population_size, replacement=True)
                if Config.search_reduce_to_mean:
                    elite_population = torch.mean(elite_population, dim=0).repeat(self.n_elite, 1)
                new_population = []
                for index in new_indices:
                    sample = elite_population[index]
                    new_population.append(
                        sample + self.sigma * torch.randn_like(sample, device=self.device)
                    )
                population = torch.stack(new_population)
            prev_mean_elite_reward = mean_elite_reward.cpu().numpy()
        return best_program, converged, num_evaluations
