import numpy as np
from karel.world import World, STATE_TABLE
from .task import Task


class Snake(Task):
        
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
        
        self.initial_body_size = 2
        self.body_size = self.initial_body_size
        self.body_list = [(self.initial_agent_y, self.initial_agent_x)]
        
        # TODO: check if marker position is not on body
        
        valid_loc = False
        while not valid_loc:
            ym = self.rng.randint(1, self.env_height - 1)
            xm = self.rng.randint(1, self.env_width - 1)
            if not state[ym, xm, 1]:
                valid_loc = True
                state[ym, xm, 6] = True
                state[ym, xm, 5] = False
                self.initial_marker_position = [ym, xm]
            
        self.marker_position = self.initial_marker_position.copy()
        
        return World(state)
    
    def reset_state(self) -> None:
        super().reset_state()
        self.body_size = self.initial_body_size
        self.body_list = [(self.initial_agent_y, self.initial_agent_x)]
        self.marker_position = self.initial_marker_position.copy()

    def get_reward(self, world_state: World):

        terminated = False
        reward = 0.
        
        # Update body and check if it reached marker
        [agent_y, agent_x, _] = world_state.get_hero_loc()
        if (agent_y == self.marker_position[0]) and (agent_x == self.marker_position[1]):
            self.body_size += 1
            self.state.s[self.marker_position[0], self.marker_position[1], 6] = False
            self.state.s[self.marker_position[0], self.marker_position[1], 5] = True
            reward = 1 / 20
            if self.body_size == 20 + self.initial_body_size:
                terminated = True
            else:
                valid_loc = False
                while not valid_loc:
                    ym = self.rng.randint(1, self.env_height - 1)
                    xm = self.rng.randint(1, self.env_width - 1)
                    if not self.state.s[ym, xm, 1] and not self.state.s[ym, xm, 4]:
                        valid_loc = True
                        self.state.s[ym, xm, 6] = True
                        self.state.s[ym, xm, 5] = False
                        self.marker_position = [ym, xm]
            
        if (agent_y, agent_x) not in self.body_list:
            last_y, last_x = self.body_list[-1]
            self.state.s[last_y, last_x, 4] = True
            self.body_list.append((agent_y, agent_x))
            if len(self.body_list) > self.body_size:
                first_y, first_x = self.body_list.pop(0)
                self.state.s[first_y, first_x, 4] = False
        
        return terminated, reward