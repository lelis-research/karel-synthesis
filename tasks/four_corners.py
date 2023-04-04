import numpy as np
from karel.world import World, STATE_TABLE
from .task import Task


class FourCorners(Task):
        
    def generate_state(self):
        
        state = np.zeros((self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool)
        
        state[:, 0, 4] = True
        state[:, self.env_width - 1, 4] = True
        state[0, :, 4] = True
        state[self.env_height - 1, :, 4] = True
        
        agent_x = self.rng.randint(2, self.env_width - 2)
        agent_pos = [self.env_height - 2, agent_x]
        
        state[agent_pos[0], agent_pos[1], 1] = True
        
        self.goal_markers = [
            [1, 1],
            [self.env_height - 2, 1],
            [1, self.env_width - 2],
            [self.env_height - 2, self.env_width - 2]
        ]
        
        self.num_previous_correct_markers = 0
        
        return World(state)
    
    def reset_state(self) -> None:
        super().reset_state()
        self.num_previous_correct_markers = 0

    def get_reward(self, world_state: World):

        terminated = False

        num_placed_markers = world_state.markers_grid.sum()
        num_correct_markers = 0

        for marker in self.goal_markers:
            if world_state.markers_grid[marker[0], marker[1]]:
                num_correct_markers += 1
        
        reward = (num_correct_markers - self.num_previous_correct_markers) / len(self.goal_markers)

        if num_placed_markers > num_correct_markers:
            terminated = True
            reward = -1        
        
        elif num_correct_markers == len(self.goal_markers):
            terminated = True
            
        self.num_previous_correct_markers = num_correct_markers
        
        return terminated, reward