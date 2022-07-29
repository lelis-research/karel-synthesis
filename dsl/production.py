from __future__ import annotations
from .base import *
import copy, random


def _get_token_list_from_node(node: Node) -> list[str]:

    if node.__class__ == ConstIntNode:
        return [f'R={str(node.value)}']
    if node.__class__ in TerminalNode.__subclasses__():
        return [node.name]

    if node.__class__ == While:
        return ['WHILE', 'c(', 'c)', 'w(', 'w)']
    if node.__class__ == Repeat:
        return ['REPEAT', 'r(', 'r)']
    if node.__class__ == If:
        return ['IF', 'c(', 'c)', 'i(', 'i)']
    if node.__class__ == ITE:
        return ['IFELSE', 'c(', 'c)', 'i(', 'i)', 'ELSE', 'e(', 'e)']
    if node.__class__ == Conjunction:
        return []

    if node.__class__ == Not:
        return ['not', 'c(', 'c)']
    if node.__class__ == And:
        return ['and', 'c(', 'c)']
    if node.__class__ == Or:
        return ['or', 'c(', 'c)']

    return []

def _get_node_from_token(token: str) -> Node:

    if token == 'move': return Move()
    if token == 'turnLeft': return TurnLeft()
    if token == 'turnRight': return TurnRight()
    if token == 'putMarker': return PutMarker()
    if token == 'pickMarker': return PickMarker()

    if token == 'frontIsClear': return FrontIsClear()
    if token == 'leftIsClear': return LeftIsClear()
    if token == 'rightIsClear': return RightIsClear()
    if token == 'markersPresent': return MarkersPresent()
    if token == 'noMarkersPresent': return NoMarkersPresent()
    
    if token == 'WHILE': return While()
    if token == 'REPEAT': return Repeat()
    if token == 'IF': return If()
    if token == 'IFELSE': return ITE()
    
    if token == 'not': return Not()
    if token == 'and': return And()
    if token == 'or': return Or()

    return None


class Production:

    def __init__(self, nodes: list[Node] = None, tokens: list[str] = None):
        self.nodes = nodes
        self.tokens = tokens

    @classmethod
    def from_nodes(cls, nodes):
        tokens = ['DEF', 'run', 'm(', 'm)']
        for node in nodes:
            tokens += _get_token_list_from_node(node)
        tokens = list(dict.fromkeys(tokens)) # Remove duplicates
        return cls(nodes, tokens)

    @classmethod
    def from_tokens(cls, tokens):
        nodes = [Conjunction()]
        for token in tokens:
            node = _get_node_from_token(token)
            if node is not None:
                nodes.append(node)
        return cls(nodes, tokens)

    @classmethod
    def from_all_nodes(cls, const_bools: list[bool], const_ints: list[int],
                       nodes_to_exclude: list[type[Node]] = None):
        nodes = [cls for cls in IntNode.__subclasses__() if cls != ConstIntNode] \
            + [cls for cls in BoolNode.__subclasses__() if cls != ConstBoolNode] \
            + [cls for cls in StatementNode.__subclasses__()] \
            + [ConstBoolNode.new(v) for v in const_bools] \
            + [ConstIntNode.new(v) for v in const_ints]
        if nodes_to_exclude is not None:
            nodes = [cls for cls in nodes if cls not in nodes_to_exclude]
        return Production.from_nodes(nodes)

    @classmethod
    def default_karel_production(cls):
        return Production.from_nodes(
            [ConstIntNode.new(i) for i in range(20)] +
            [
                While(), Repeat(), If(), ITE(), Conjunction(), Not(), FrontIsClear(),
                RightIsClear(), LeftIsClear(), MarkersPresent(), NoMarkersPresent(),
                Move(), TurnLeft(), TurnRight(), PickMarker(), PutMarker()
            ]
        )

    def get_actions(self):
        return [
            n for n in self.nodes
            if n.__class__ in StatementNode.__subclasses__()
            and n.__class__ in TerminalNode.__subclasses__()
        ]

    def _fill_random_program(self, node: Node, depth: int, max_depth: int) -> None:
        for i in range(node.get_number_children()):
            child_type = node.get_children_types()[i]
            prod_list = [obj for obj in self.nodes if obj.__class__ in child_type.__subclasses__()]
            if depth >= max_depth:
                prod_list = [
                    obj for obj in self.nodes 
                    if obj.__class__ in child_type.__subclasses__()
                    and obj.__class__ in TerminalNode.__subclasses__()
                ]
            node_id = random.randint(0, len(prod_list) - 1)
            child = copy.deepcopy(prod_list[node_id])
            node.add_child(child)
            self._fill_random_program(child, depth + 1, max_depth)

    # TODO: this random program generation is different from LEAPS's because it needs Conjunction
    # objects to concatenate expressions, which results in "more nested" programs instead of "more
    # sequential" programs. Maybe I should create both methods
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