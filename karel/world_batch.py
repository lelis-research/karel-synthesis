import numpy as np
from karel.world import World


class WorldBatch:

    def __init__(self, states: np.ndarray):
        self.worlds: list[World] = []
        for s in states:
            self.worlds.append(World(s))

    def step(self, actions):
        assert len(self.worlds) == len(actions)
        for w, a in zip(self.worlds, actions):
            if a < 5: # Action 5 is the "do nothing" action, for filling up empty space in the array
                w.run_action(a)
        return np.array([w.get_state() for w in self.worlds])