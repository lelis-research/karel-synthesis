import numpy as np
from config import Config
from karel.world import STATE_TABLE, World


class WorldGenerator:

    def __init__(self) -> None:
        self.rng = np.random.RandomState(Config.env_seed)
        self.h = Config.env_height
        self.w = Config.env_width

    def generate(self, wall_prob=0.1, marker_prob=0.1) -> World:
        s = np.zeros((self.h, self.w, len(STATE_TABLE)), dtype=bool)
        # Wall
        s[:, :, 4] = self.rng.rand(self.h, self.w) > 1 - wall_prob
        s[0, :, 4] = True
        s[self.h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, self.w-1, 4] = True
        # Karel initial location
        valid_loc = False
        while(not valid_loc):
            y = self.rng.randint(0, self.h)
            x = self.rng.randint(0, self.w)
            if not s[y, x, 4]:
                valid_loc = True
                s[y, x, self.rng.randint(0, 4)] = True
        # Marker: num of max marker == 1 for now TODO: this is the setting for LEAPS - do we keep it?
        s[:, :, 6] = (self.rng.rand(self.h, self.w) > 1 - marker_prob) * (s[:, :, 4] == False) > 0
        s[:, :, 5] = np.sum(s[:, :, 6:], axis=-1) == 0
        return World(s)
