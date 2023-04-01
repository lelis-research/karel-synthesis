import numpy as np
from karel.world import World, STATE_TABLE
from .clean_house import CleanHouse


class CleanHouseSparse(CleanHouse):
        
    def get_reward(self, world_state: World):

        terminated = False
        reward = 0
        
        num_markers = world_state.markers_grid.sum()
        
        if num_markers > self.previous_number_of_markers:
            reward = -1
            terminated = True
        
        elif num_markers == 0:
            reward = 1
            terminated = True
        
        return terminated, reward