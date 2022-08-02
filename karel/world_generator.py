import numpy as np
from karel.world import World


class WorldGenerator:

    def __init__(self, seed) -> None:
        self.rng = np.random.RandomState(seed)

    def generate(self, h=8, w=8, wall_prob=0.1, marker_prob=0.1) -> World:
        s = np.zeros((h, w, 16), dtype=bool)
        # Wall
        s[:, :, 4] = self.rng.rand(h, w) > 1 - wall_prob
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True
        # Karel initial location
        valid_loc = False
        while(not valid_loc):
            y = self.rng.randint(0, h)
            x = self.rng.randint(0, w)
            if not s[y, x, 4]:
                valid_loc = True
                s[y, x, self.rng.randint(0, 4)] = True
        # Marker: num of max marker == 1 for now TODO: this is the setting for LEAPS - do we keep it?
        s[:, :, 6] = (self.rng.rand(h, w) > 1 - marker_prob) * (s[:, :, 4] == False) > 0
        s[:, :, 5] = np.sum(s[:, :, 6:], axis=-1) == 0
        return World(s)
