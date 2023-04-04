import numpy as np
from karel.world import World, STATE_TABLE
from .task import Task


class TopOff(Task):
        
    def generate_state(self):
        
        state = np.zeros((self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool)
        
        state[:, 0, 4] = True
        state[:, self.env_width - 1, 4] = True
        state[0, :, 4] = True
        state[self.env_height - 1, :, 4] = True
        
        state[self.env_height - 2, 1, 1] = True
        
        self.possible_marker_locations = [
            [self.env_height - 2, i] for i in range(2, self.env_width - 1)
        ]
        
        self.rng.shuffle(self.possible_marker_locations)
        
        self.num_markers = self.rng.randint(1, len(self.possible_marker_locations))
        self.markers = self.possible_marker_locations[:self.num_markers]
        
        for marker in self.markers:
            state[marker[0], marker[1], 6] = True
        
        self.num_previous_correct_markers = 0
        
        return World(state)
    
    def reset_state(self) -> None:
        super().reset_state()
        self.num_previous_correct_markers = 0

    def get_reward(self, world_state: World):

        terminated = False
        
        num_markers = world_state.markers_grid.sum()
        num_correct_markers = 0

        for marker in self.markers:
            if world_state.markers_grid[marker[0], marker[1]] == 2:
                num_correct_markers += 1
            elif world_state.markers_grid[marker[0], marker[1]] == 0:
                return True, -1
        
        reward = (num_correct_markers - self.num_previous_correct_markers) / len(self.markers)
        
        if num_markers > num_correct_markers + len(self.markers):
            terminated = True
            reward = -1
        
        elif num_correct_markers == len(self.markers):
            terminated = True
            
        self.num_previous_correct_markers = num_correct_markers
        
        return terminated, reward