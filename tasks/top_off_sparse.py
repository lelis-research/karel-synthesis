import numpy as np
from karel.world import World, STATE_TABLE
from .top_off import TopOff


class TopOffSparse(TopOff):

    def get_reward(self, world_state: World):

        terminated = False
        num_correct_markers = 0
        reward = 0

        for marker in self.markers:
            if world_state.markers_grid[marker[0], marker[1]] == 2:
                num_correct_markers += 1
            elif world_state.markers_grid[marker[0], marker[1]] == 0:
                return True, -1
        
        num_markers = world_state.markers_grid.sum()
        if num_markers > num_correct_markers + len(self.markers):
            terminated = True
            reward = -1
        
        if num_correct_markers == len(self.markers):
            terminated = True
            reward = 1
        
        return terminated, reward