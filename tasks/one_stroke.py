import numpy as np
from karel.world import World, STATE_TABLE
from .task import Task


class OneStroke(Task):
        
    def generate_state(self):
        
        state = np.zeros((self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool)
        
        state[:, 0, 4] = True
        state[:, self.env_width - 1, 4] = True
        state[0, :, 4] = True
        state[self.env_height - 1, :, 4] = True
        
        self.initial_agent_x = self.rng.randint(1, self.env_width - 1)
        self.initial_agent_y = self.rng.randint(1, self.env_height - 1)
        
        state[self.initial_agent_y, self.initial_agent_x, 1] = True
        
        state[:, :, 5] = True
        
        self.prev_agent_x = self.initial_agent_x
        self.prev_agent_y = self.initial_agent_y
        self.number_cells_visited = 0
        self.max_number_cells = (self.env_width - 2) * (self.env_height - 2)
        
        return World(state)
    
    def reset_state(self) -> None:
        super().reset_state()
        self.prev_agent_x = self.initial_agent_x
        self.prev_agent_y = self.initial_agent_y
        self.number_cells_visited = 0

    def get_reward(self, world_state: World):

        terminated = False
        
        [agent_y, agent_x, _] = world_state.get_hero_loc()
        
        if (agent_x == self.prev_agent_x) and (agent_y == self.prev_agent_y):
            reward = 0
        else:
            self.number_cells_visited += 1
            reward = 1 / self.max_number_cells
            # Place a wall where the agent was
            world_state.s[self.prev_agent_y, self.prev_agent_x, 4] = True

        if self.number_cells_visited == self.max_number_cells:
            terminated = True
        
        self.prev_agent_x = agent_x
        self.prev_agent_y = agent_y
        
        return terminated, reward