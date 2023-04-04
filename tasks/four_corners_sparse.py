import numpy as np
from karel.world import World, STATE_TABLE
from .four_corners import FourCorners


class FourCornersSparse(FourCorners):

    def get_reward(self, world_state: World):

        terminated = False
        reward = 0

        num_placed_markers = world_state.markers_grid.sum()
        num_correct_markers = 0

        for marker in self.goal_markers:
            if world_state.markers_grid[marker[0], marker[1]]:
                num_correct_markers += 1

        if num_placed_markers > num_correct_markers:
            terminated = True
            reward = -1        
        
        elif num_correct_markers == len(self.goal_markers):
            terminated = True
            reward = 1
        
        return terminated, reward