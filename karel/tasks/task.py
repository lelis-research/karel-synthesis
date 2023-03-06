from abc import ABC, abstractmethod
import numpy as np

from config.config import Config


class Task(ABC):
    
    def __init__(self, config: Config):
        self.rng = np.random.RandomState(config.env_seed)
        self.env_height = config.env_height
        self.env_width = config.env_width
    
    @abstractmethod
    def generate_state(self):
        pass
    
    @abstractmethod
    def get_reward(self):
        pass