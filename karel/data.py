import numpy as np
from karel.world import World
import json


class Data: # TODO: rename

    def __init__(self, inputs, targets = None):
        self.inputs = inputs
        self.targets = targets

    @classmethod
    def from_matrix(cls, state: np.ndarray):
        return cls(state) # TODO

    @classmethod
    def from_json(cls, file_name, line_number):
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                if i == line_number - 1:
                    data = json.loads(line)
                    break

        inputs = []
        targets = []

        for example in data['examples']:
            inputs.append(World.from_json(example['inpgrid_json']))
            targets.append(World.from_json(example['outgrid_json']))

        return cls(inputs, targets)

    def step(self, actions):
        pass #TODO: exec actions in all inputs