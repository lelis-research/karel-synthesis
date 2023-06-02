import numpy as np
from karel.world import World, STATE_TABLE
from .seeder import Seeder


class SeederSparse(Seeder):

    def get_reward(self, world_state: World):

        reward = 0        
        terminated = False
        num_markers = world_state.markers_grid.sum()
        
        if (world_state.markers_grid > 1).any():
            reward = -1
            terminated = True
        
        elif num_markers < self.previous_number_of_markers:
            reward = -1
            terminated = True
        
        elif num_markers == self.max_number_of_markers:
            reward = 1
            terminated = True
        
        return terminated, reward