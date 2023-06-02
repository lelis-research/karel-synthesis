import numpy as np
from scipy import spatial
from karel.world import World, STATE_TABLE
from .task import Task


class FindMarker(Task):
        
    def generate_state(self):
        
        state = np.zeros((self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool)
        
        state[:, 0, 4] = True
        state[:, self.env_width - 1, 4] = True
        state[0, :, 4] = True
        state[self.env_height - 1, :, 4] = True
        
        self.marker_position = [self.rng.randint(1, self.env_height - 1), self.rng.randint(1, self.env_width - 1)]
        
        state[:, :, 5] = True
        state[self.marker_position[0], self.marker_position[1], 6] = True
        state[self.marker_position[0], self.marker_position[1], 5] = False
        
        valid_loc = False
        while not valid_loc:
            init_pos = [self.rng.randint(1, self.env_height - 1), self.rng.randint(1, self.env_width - 1)]
            if not state[init_pos[0], init_pos[1], 6]:
                valid_loc = True
                state[init_pos[0], init_pos[1], 1] = True
        
        self.initial_distance = spatial.distance.cityblock(init_pos, self.marker_position)
        self.previous_distance = self.initial_distance
        
        return World(state)

    def reset_state(self) -> None:
        super().reset_state()
        self.previous_distance = self.initial_distance

    def get_reward(self, world_state: World):

        terminated = False
        reward = 0.
        num_markers = world_state.markers_grid.sum()

        if num_markers > 1:
            terminated = True
            reward = -1
        else:
            karel_pos = world_state.get_hero_loc()
            current_distance = spatial.distance.cityblock([karel_pos[0], karel_pos[1]], self.marker_position)
            reward = (self.previous_distance - current_distance) / self.initial_distance
            self.previous_distance = current_distance
        
        # Check if marker has been picked up
        if world_state.markers_grid[self.marker_position[0], self.marker_position[1]] == 0:
            terminated = True
        
        return terminated, reward