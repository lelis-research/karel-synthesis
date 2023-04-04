import numpy as np
from karel.world import World, STATE_TABLE
from .task import Task


class Harvester(Task):
        
    def generate_state(self):
        
        state = np.zeros((self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool)
        
        state[:, 0, 4] = True
        state[:, self.env_width - 1, 4] = True
        state[0, :, 4] = True
        state[self.env_height - 1, :, 4] = True
        
        agent_x = self.rng.randint(1, self.env_width - 1)
        agent_pos = [self.env_height - 2, agent_x]
        
        state[agent_pos[0], agent_pos[1], 1] = True
        
        state[1:self.env_height - 1, 1:self.env_width - 1, 6] = True
        
        self.initial_number_of_markers = state[:, :, 6].sum()
        self.previous_number_of_markers = self.initial_number_of_markers
        
        return World(state)
    
    def reset_state(self) -> None:
        super().reset_state()
        self.previous_number_of_markers = self.initial_number_of_markers

    def get_reward(self, world_state: World):

        terminated = False
        
        num_markers = world_state.markers_grid.sum()
        
        reward = (self.previous_number_of_markers - num_markers) / self.initial_number_of_markers
        
        if num_markers > self.previous_number_of_markers:
            reward = -1
            terminated = True
        
        elif num_markers == 0:
            terminated = True
        
        self.previous_number_of_markers = num_markers
        
        return terminated, reward