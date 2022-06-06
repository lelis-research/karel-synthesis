from .world import World
import json


class Data:

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

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

        # This class is a single set of inputs and outputs (single task)
        # Then, create a class of Dataset to train different tasks