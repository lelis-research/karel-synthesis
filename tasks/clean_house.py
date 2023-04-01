import numpy as np
from scipy import spatial
from karel.world import World, STATE_TABLE
from .task import Task


class CleanHouse(Task):
        
    def generate_state(self):
        
        world_map = [
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            ['-',   0,   0, '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',   0, '-',   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-', '-', '-', '-',   0, '-', '-'],
            ['-', '-',   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-', '-',   0,   0,   0,   0,   0,   0,   0,   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-', '-', '-',   0, '-',   0, '-', '-', '-',   0, '-',   0,   0, '-', '-', '-',   0, '-',   0, '-', '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-', '-',   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ]
        
        assert self.env_height == 14 and self.env_width == 22
        
        state = np.zeros((self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool)
        
        agent_pos = (1, 13)
        hardcoded_invalid_marker_locations = [(1, 13), (2, 12), (3, 10), (4, 11), (5, 11), (6, 10)]
        state[agent_pos[0], agent_pos[1], 2] = True
        
        state[:, :, 5] = True
        possible_marker_locations = []
        
        for y1 in range(self.env_height):
            for x1 in range(self.env_width):
                if world_map[y1][x1] == '-':
                    state[y1, x1, 4] = True
                else:
                    if (y1, x1) not in hardcoded_invalid_marker_locations:
                        possible_marker_locations.append([y1, x1])
        
        self.rng.shuffle(possible_marker_locations)
        
        for marker_location in possible_marker_locations[:10]:
            state[marker_location[0], marker_location[1], 5] = False
            state[marker_location[0], marker_location[1], 6] = True
        
        # put 2 markers near start point for end condition
        state[agent_pos[0]+1, agent_pos[1]-1, 5] = False
        state[agent_pos[0]+1, agent_pos[1]-1, 7] = True
        
        self.initial_number_of_markers = state[:, :, 6].sum() + 2 * state[:, :, 7].sum()
        self.previous_number_of_markers = self.initial_number_of_markers
        
        return World(state)

    def get_reward(self, world_state: World):

        terminated = False
        
        num_markers = world_state.markers_grid.sum()
        
        reward = (self.previous_number_of_markers - num_markers) / self.initial_number_of_markers
        
        if num_markers > self.previous_number_of_markers:
            reward = -1
            terminated = True
        
        elif num_markers == 0:
            terminated = True
        
        self.previous_number_of_markers = num_markers
        
        return terminated, reward