from math import ceil
import numpy as np
from karel.world import World, STATE_TABLE
from .task import Task


class DoorKey(Task):
        
    def generate_state(self):
        
        state = np.zeros((self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool)
        
        state[:, 0, 4] = True
        state[:, self.env_width - 1, 4] = True
        state[0, :, 4] = True
        state[self.env_height - 1, :, 4] = True
        
        wall_column = ceil(self.env_width / 2)
        state[:, wall_column, 4] = True
        
        self.key_cell = (self.rng.randint(1, self.env_height - 1), self.rng.randint(1, wall_column))
        self.end_marker_cell = (self.rng.randint(1, self.env_height - 1), self.rng.randint(wall_column + 1, self.env_width - 1))
        
        state[:, :, 5] = True
        state[self.key_cell[0], self.key_cell[1], 6] = True
        state[self.key_cell[0], self.key_cell[1], 5] = False
        state[self.end_marker_cell[0], self.end_marker_cell[1], 6] = True
        state[self.end_marker_cell[0], self.end_marker_cell[1], 5] = False
        
        valid_loc = False
        while not valid_loc:
            y_agent = self.rng.randint(1, self.env_height - 1)
            x_agent = self.rng.randint(1, wall_column)
            if not state[y_agent, x_agent, 6]:
                valid_loc = True
                state[y_agent, x_agent, 1] = True

        self.door_cells = [(2, wall_column), (3, wall_column)]
        self.door_locked = True
        
        return World(state)
    
    def reset_state(self) -> None:
        super().reset_state()
        self.door_locked = True

    def get_reward(self, world_state: World):

        terminated = False
        reward = 0.
        num_markers = world_state.markers_grid.sum()
        
        if self.door_locked:
            if num_markers > 2:
                terminated = True
                reward = -1
            # Check if key has been picked up
            elif world_state.markers_grid[self.key_cell[0], self.key_cell[1]] == 0:
                self.door_locked = False
                for door_cell in self.door_cells:
                    world_state.s[door_cell[0], door_cell[1], 4] = False
                reward = 0.5
        else:
            if num_markers > 1:
                # Check if end marker has been topped off
                if world_state.markers_grid[self.end_marker_cell[0], self.end_marker_cell[1]] == 2:
                    terminated = True
                    reward = 0.5
                else:
                    terminated = True
                    reward = -1
            elif num_markers == 0:
                terminated = True
                reward = -1
        
        return terminated, reward