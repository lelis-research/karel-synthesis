from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from typing import Union
import numpy as np

from config.config import Config
from karel.world import World


class Task(ABC):
    
    def __init__(self, seed: Union[None, int] = None):
        if seed is None:
            self.rng = np.random.RandomState(Config.env_seed)
        else:
            self.rng = np.random.RandomState(seed)
        self.env_height = Config.env_height
        self.env_width = Config.env_width
        self.state = self.generate_state()
        self.initial_state = copy.deepcopy(self.state)
    
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