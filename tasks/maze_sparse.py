import numpy as np
from karel.world import World, STATE_TABLE
from .maze import Maze


class MazeSparse(Maze):

    def get_reward(self, world_state: World):

        terminated = False
        reward = 0

        karel_pos = world_state.get_hero_loc()
            
        if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
            reward = 1
            terminated = True
        
        return terminated, reward