from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union
import numpy as np

from config.config import Config
from karel.world import World


class Task(ABC):
    
    def __init__(self):
        self.rng = np.random.RandomState(Config.env_seed)
        self.env_height = Config.env_height
        self.env_width = Config.env_width
    
    @abstractmethod
    def generate_state(self, seed: Union[None, int] = None) -> World:
        pass
    
    @abstractmethod
    def get_reward(self, world_state: World) -> tuple[bool, float]:
        pass