import numpy as np
from scipy import spatial
from karel.world import World, STATE_TABLE
from .task import Task


class Maze(Task):
        
    def generate_state(self):
        
        def get_neighbors(cur_pos):
            neighbor_list = []
            #neighbor top
            if cur_pos[0] - 2 > 0: neighbor_list.append([cur_pos[0] - 2, cur_pos[1]])
            # neighbor bottom
            if cur_pos[0] + 2 < self.env_height - 1: neighbor_list.append([cur_pos[0] + 2, cur_pos[1]])
            # neighbor left
            if cur_pos[1] - 2 > 0: neighbor_list.append([cur_pos[0], cur_pos[1] - 2])
            # neighbor right
            if cur_pos[1] + 2 < self.env_width - 1: neighbor_list.append([cur_pos[0], cur_pos[1] + 2])
            return neighbor_list
        
        state = np.zeros((self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool)
        state[:, :, 4] = True
        
        init_pos = [self.env_height - 2, 1]
        state[init_pos[0], init_pos[1], 1] = True
        state[init_pos[0], init_pos[1], 4] = False
        visited = np.zeros((self.env_height, self.env_width), dtype=bool)
        visited[init_pos[0], init_pos[1]] = True
        
        stack = [init_pos]
        while len(stack) > 0:
            cur_pos = stack.pop()
            neighbors = get_neighbors(cur_pos)
            self.rng.shuffle(neighbors)
            for neighbor in neighbors:
                if not visited[neighbor[0], neighbor[1]]:
                    visited[neighbor[0], neighbor[1]] = True
                    state[(cur_pos[0] + neighbor[0]) // 2, (cur_pos[1] + neighbor[1]) // 2, 4] = False
                    state[neighbor[0], neighbor[1], 4] = False
                    stack.append(neighbor)
        
        valid_loc = False
        state[:, :, 5] = True
        while not valid_loc:
            ym = self.rng.randint(1, self.env_height - 1)
            xm = self.rng.randint(1, self.env_width - 1)
            if not state[ym, xm, 4] and not state[ym, xm, 1]:
                valid_loc = True
                state[ym, xm, 6] = True
                state[ym, xm, 5] = False
                self.marker_position = [ym, xm]
        
        self.initial_distance = spatial.distance.cityblock(init_pos, self.marker_position)
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

        if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
            terminated = True
        
        self.previous_distance = current_distance
        
        return terminated, reward