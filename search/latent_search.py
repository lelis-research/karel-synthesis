from __future__ import annotations
from dsl.parser import Parser
from embedding.autoencoder.base_vae import BaseVAE
from logger.stdout_logger import StdoutLogger
from tasks.task import Task
from config.config import Config
import torch


class LatentSearch:
    
    def __init__(self, model: BaseVAE, task: Task):
        self.model = model
        self.device = self.model.device
        self.population_size = Config.search_population_size
        self.elitism_rate = Config.search_elitism_rate
        self.n_elite = int(Config.search_elitism_rate * self.population_size)
        self.number_executions = Config.search_number_executions
        self.number_iterations = Config.search_number_iterations
        self.sigma = Config.search_sigma
        self.model_hidden_size = Config.model_hidden_size
        self.task = task
        
    def init_population(self):
        self.population = torch.stack([
            torch.randn(self.model_hidden_size, device=self.device) for _ in range(self.population_size)
        ])
        
    def execute_population(self) -> tuple[list[str], torch.Tensor]:
        programs_tokens = self.model.decode_vector(self.population)
        rewards = []
        programs = []
        for program_tokens in programs_tokens:
            program_str = Parser.list_to_tokens(program_tokens)
            programs.append(program_str)
            try:
                program = Parser.tokens_to_nodes(program_str)
            except AssertionError: # Invalid program
                mean_reward = -1
                rewards.append(mean_reward)
                continue
            mean_reward = 0.
            for seed in range(self.number_executions):
                state = self.task.generate_state(seed)
                reward = 0
                for _ in program.run_generator(state):
                    terminated, instant_reward = self.task.get_reward(state)
                    reward += instant_reward
                    if terminated:
                        break
                mean_reward += reward
            mean_reward /= self.number_executions
            rewards.append(mean_reward)
        return programs, torch.tensor(rewards, device=self.device)
    
    def search(self):
        self.init_population()
        for iteration in range(1, self.number_iterations + 1):
            programs, rewards = self.execute_population()
            best_indices = torch.topk(rewards, self.n_elite).indices
            elite_population = self.population[best_indices]
            mean_elite_reward = torch.mean(rewards[best_indices])
            best_program = programs[torch.argmax(rewards)]
            best_reward = torch.max(rewards)
            StdoutLogger.log('Latent Search',f'Iteration {iteration}.')
            StdoutLogger.log('Latent Search',f'Mean elite reward: {mean_elite_reward}')
            StdoutLogger.log('Latent Search',f'Best reward: {best_reward}')
            StdoutLogger.log('Latent Search',f'Best program: {best_program}')
            if mean_elite_reward.cpu().numpy() == 1.0:
                break
            new_indices = torch.ones(elite_population.size(0), device=self.device).multinomial(
                self.population_size, replacement=True)
            new_population = []
            for index in new_indices:
                sample = elite_population[index]
                new_population.append(
                    sample + self.sigma * torch.randn_like(sample, device=self.device)
                )
            self.population = torch.stack(new_population)
        return best_program