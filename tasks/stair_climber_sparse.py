import numpy as np
from karel.world import World, STATE_TABLE
from .stair_climber import StairClimber


class StairClimberSparse(StairClimber):

    def get_reward(self, world_state: World):

        terminated = False
        reward = 0

        karel_pos = world_state.get_hero_loc()
        
        if [karel_pos[0], karel_pos[1]] not in self.valid_positions:
            reward = -1
            terminated = True
            
        if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
            reward = 1
            terminated = True
        
        return terminated, reward