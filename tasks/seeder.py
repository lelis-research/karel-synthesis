import numpy as np
from karel.world import World, STATE_TABLE
from .task import Task


class Seeder(Task):
        
    def generate_state(self):
        
        state = np.zeros((self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool)
        
        state[:, 0, 4] = True
        state[:, self.env_width - 1, 4] = True
        state[0, :, 4] = True
        state[self.env_height - 1, :, 4] = True
        
        agent_x = self.rng.randint(1, self.env_width - 1)
        agent_y = self.rng.randint(1, self.env_height - 1)
        
        state[agent_y, agent_x, 1] = True
        
        state[:, :, 5] = True
        
        self.max_number_of_markers = (self.env_width - 2) * (self.env_height - 2)
        self.previous_number_of_markers = 0
        
        return World(state)
    
    def reset_state(self) -> None:
        super().reset_state()
        self.previous_number_of_markers = 0

    def get_reward(self, world_state: World):

        terminated = False
        
        num_markers = world_state.markers_grid.sum()
        
        reward = (num_markers - self.previous_number_of_markers) / self.max_number_of_markers
        
        if (world_state.markers_grid > 1).any():
            reward = -1
            terminated = True
        
        elif num_markers < self.previous_number_of_markers:
            reward = -1
            terminated = True
        
        elif num_markers == self.max_number_of_markers:
            terminated = True
        
        self.previous_number_of_markers = num_markers
        
        return terminated, reward