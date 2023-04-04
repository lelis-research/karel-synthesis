import numpy as np
from scipy import spatial
from karel.world import World, STATE_TABLE
from .task import Task


class StairClimber(Task):
        
    def generate_state(self):
        
        state = np.zeros((self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool)
        
        state[:, 0, 4] = True
        state[:, self.env_width - 1, 4] = True
        state[0, :, 4] = True
        state[self.env_height - 1, :, 4] = True
        
        for i in range(1, self.env_width - 2):
            state[self.env_height - i - 1, i + 1, 4] = True
            state[self.env_height - i - 1, i + 2, 4] = True
        
        on_stair_positions = [
            [self.env_height - i - 1, i] for i in range(1, self.env_width - 1)
        ]
        
        one_block_above_stair_positions = [
            [self.env_height - i - 2, i] for i in range(1, self.env_width - 2)
        ]
        
        # One cell above the stairs
        self.valid_positions = on_stair_positions + one_block_above_stair_positions
        
        # Initial position has to be on stair but cannot be on last step
        initial_position_index = self.rng.randint(0, len(on_stair_positions) - 1)
        
        # Marker has to be after initial position
        marker_position_index = self.rng.randint(initial_position_index + 1, len(on_stair_positions))
        
        self.initial_position = on_stair_positions[initial_position_index]
        state[self.initial_position[0], self.initial_position[1], 1] = True
        
        self.marker_position = on_stair_positions[marker_position_index]
        state[:, :, 5] = True
        state[self.marker_position[0], self.marker_position[1], 6] = True
        state[self.marker_position[0], self.marker_position[1], 5] = False
        
        self.initial_distance = spatial.distance.cityblock(self.initial_position, self.marker_position)
        self.previous_distance = self.initial_distance
        
        return World(state)
    
    def reset_state(self) -> None:
        super().reset_state()
        self.previous_distance = self.initial_distance

    def get_reward(self, world_state: World):

        terminated = False
        reward = 0

        karel_pos = world_state.get_hero_loc()
        
        current_distance = spatial.distance.cityblock([karel_pos[0], karel_pos[1]], self.marker_position)
        
        # Reward is how much closer Karel is to the marker, normalized by the initial distance
        reward = (self.previous_distance - current_distance) / self.initial_distance
        
        if [karel_pos[0], karel_pos[1]] not in self.valid_positions:
            reward = -1
            terminated = True
            
        if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
            terminated = True
        
        self.previous_distance = current_distance
        
        return terminated, reward