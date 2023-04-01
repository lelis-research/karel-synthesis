from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from typing import Union
import numpy as np

from config import Config
from dsl.base import Program
from karel.world import World


class Task(ABC):
    
    def __init__(self, seed: Union[None, int] = None):
        if seed is None:
            self.rng = np.random.RandomState(Config.env_seed)
        else:
            self.rng = np.random.RandomState(seed)
        self.env_height = Config.env_height
        self.env_width = Config.env_width
        self.initial_state = self.generate_state()
        self.reset_state()
    
    def get_state(self) -> World:
        return self.state
    
    def reset_state(self) -> None:
        self.state = copy.deepcopy(self.initial_state)
    
    @abstractmethod
    def generate_state(self) -> World:
        pass
    
    @abstractmethod
    def get_reward(self, world_state: World) -> tuple[bool, float]:
        pass
    
    def evaluate_program(self, program: Program) -> float:
        self.reset_state()
        reward = 0
        steps = 0
        for _ in program.run_generator(self.state):
            terminated, instant_reward = self.get_reward(self.state)
            reward += instant_reward
            steps += 1
            if terminated or steps >= Config.data_max_demo_length:
                break
        return reward
