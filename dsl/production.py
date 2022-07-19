from .base import *
import copy, random

# TODO: export tokens and implement intseq2str

class Production:

    def __init__(self, operations: list = None, terminals: list = None):
        self.operations = operations
        self.terminals = terminals
        self.all_nodes = self.operations + self.terminals

    @classmethod
    def from_all_nodes(cls, const_bools: list[bool], const_ints: list[int],
                       nodes_to_exclude: list = None):
        nodes = [cls for cls in IntNode.__subclasses__() if cls != ConstIntNode] \
            + [cls for cls in BoolNode.__subclasses__() if cls != ConstBoolNode] \
            + [cls for cls in StatementNode.__subclasses__()]
        if nodes_to_exclude is not None:
            nodes_to_exclude = [cls for cls in nodes if cls not in nodes_to_exclude]
        operations = [cls() for cls in nodes if cls in OperationNode.__subclasses__()]
        terminals = [cls() for cls in nodes if cls in TerminalNode.__subclasses__()] \
            + [ConstBoolNode.new(v) for v in const_bools] \
            + [ConstIntNode.new(v) for v in const_ints]
        return cls(operations, terminals)

    def get_all_nodes(self):
        return [n for n in self.operations] + [n for n in self.terminals]

    def get_conditionals(self):
        return [n for n in self.terminals if n.__class__ in BoolNode.__subclasses__()]

    def get_actions(self):
        return [n for n in self.terminals if n.__class__ in StatementNode.__subclasses__()]

    @classmethod
    def default_karel_production(cls):
        const_ints = [ConstIntNode.new(i) for i in range(20)]
        operations = [While(), Repeat(), If(), ITE(), Conjunction(), Not()]
        terminals = [
            FrontIsClear(), RightIsClear(), LeftIsClear(), MarkersPresent(), NoMarkersPresent(),
            Move(), TurnLeft(), TurnRight(), PickMarker(), PutMarker()
        ]
        return cls(operations, terminals + const_ints)

    def _fill_random_program(self, node: Node, depth: int, max_depth: int) -> None:
        for i in range(node.get_number_children()):
            child_type = node.get_children_types()[i]
            prod_list = [obj for obj in self.all_nodes if obj.__class__ in child_type.__subclasses__()]
            if depth >= max_depth:
                prod_list = [obj for obj in self.terminals if obj.__class__ in child_type.__subclasses__()]
            node_id = random.randint(0, len(prod_list) - 1)
            child = copy.deepcopy(prod_list[node_id])
            node.add_child(child)
            self._fill_random_program(child, depth + 1, max_depth)

    def random_program(self, max_depth: int) -> Program:
        program = Program()
        self._fill_random_program(program, 0, max_depth)
        return program

    def get_tokens(self) -> list[str]:
        # TODO: do a proper parse based on self.all_nodes
        return [
            'DEF', 'run', 'm(', 'm)', 'move', 'turnRight', 'turnLeft', 'pickMarker', 'putMarker',
            'r(', 'r)', 'R=0', 'R=1', 'R=2', 'R=3', 'R=4', 'R=5', 'R=6', 'R=7', 'R=8', 'R=9', 'R=10',
            'R=11', 'R=12', 'R=13', 'R=14', 'R=15', 'R=16', 'R=17', 'R=18', 'R=19', 'REPEAT', 'c(',
            'c)', 'i(', 'i)', 'e(', 'e)', 'IF', 'IFELSE', 'ELSE', 'frontIsClear', 'leftIsClear',
            'rightIsClear', 'markersPresent', 'noMarkersPresent', 'not', 'w(', 'w)', 'WHILE'
        ]