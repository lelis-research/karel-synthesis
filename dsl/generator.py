import copy
import numpy as np

from config import Config
from dsl.dsl import DSL
from dsl.production import Production
from .base import *

class ProgramGenerator:
    
    # From LEAPS paper, p. 41
    valid_int_values = list(range(0, 20))
    valid_bool_values = [False, True]
    action_prob = 0.2
    bool_not_prob = 0.1
    nodes_probs = {
        StatementNode: {
            While: 0.15,
            Repeat: 0.03,
            Conjunction: 0.5,
            If: 0.08,
            ITE: 0.04,
            Move: action_prob * 0.5,
            TurnLeft: action_prob * 0.15,
            TurnRight: action_prob * 0.15,
            PickMarker: action_prob * 0.1,
            PutMarker: action_prob * 0.1
        },
        BoolNode: {
            Not: bool_not_prob,
            FrontIsClear: (1 - bool_not_prob) * 0.5,
            LeftIsClear: (1 - bool_not_prob) * 0.15,
            RightIsClear: (1 - bool_not_prob) * 0.15,
            MarkersPresent: (1 - bool_not_prob) * 0.1,
            NoMarkersPresent: (1 - bool_not_prob) * 0.1,
        },
        IntNode: {
            ConstIntNode: 1
        }
    }
    
    @staticmethod
    def get_node_probs(node_type: type[Node]) -> dict[type[Node], float]:
        return ProgramGenerator.nodes_probs[node_type]
    
    def __init__(self, dsl: DSL, seed: Union[None, int] = None) -> None:
        self.dsl = dsl
        self.max_depth = Config.datagen_max_depth
        self.max_sequential_length = Config.datagen_max_sequential_length
        self.max_program_length = Config.data_max_program_len
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(Config.env_seed)
        
    def _fill_children(self, node: Node, current_depth: int = 1, current_sequential_length: int = 0) -> None:
        node_production_rules = Production.get_production_rules(type(node))
        for i, child_type in enumerate(node.get_children_types()):
            child_probs = copy.deepcopy(ProgramGenerator.get_node_probs(child_type))
            # Masking out invalid children by production rules
            for child_type in child_probs:
                # if child_type is None:
                #     continue
                if child_type not in node_production_rules[i]:
                    child_probs[child_type] = 0
                if current_depth >= self.max_depth and child_type.get_number_children() > 0:
                    child_probs[child_type] = 0
            if issubclass(type(node), Conjunction) and current_sequential_length >= self.max_sequential_length:
                if Conjunction in child_probs:
                    child_probs[Conjunction] = 0

            p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
            child = self.rng.choice(list(child_probs.keys()), p=p_list)
            child_instance = child()
            if child.get_number_children() > 0:
                if issubclass(type(node), Conjunction):
                    self._fill_children(child_instance, current_depth + child.node_depth, current_sequential_length + 1)
                else:
                    self._fill_children(child_instance, current_depth + child.node_depth, 0)
                    
            elif child == ConstIntNode:
                child_instance.value = self.rng.choice(ProgramGenerator.valid_int_values)
            elif child == ConstBoolNode:
                child_instance.value = self.rng.choice(ProgramGenerator.valid_bool_values)

            node.children[i] = child_instance
        
    def generate_program(self) -> Program:
        while True:
            program = Program()
            self._fill_children(program)
            if len(self.dsl.parse_node_to_int(program)) <= self.max_program_length:
                break
        return program