from __future__ import annotations

from config import Config
from .base import *


def _get_token_list_from_node(node: Node) -> list[str]:

    if node is None:
        return ['<HOLE>']

    if type(node) == ConstIntNode:
        return [f'R={str(node.value)}']
    if type(node) in TerminalNode.__subclasses__():
        return [node.name]

    if type(node) == While:
        return ['WHILE', 'c(', 'c)', 'w(', 'w)']
    if type(node) == Repeat:
        return ['REPEAT', 'r(', 'r)']
    if type(node) == If:
        return ['IF', 'c(', 'c)', 'i(', 'i)']
    if type(node) == ITE:
        return ['IFELSE', 'c(', 'c)', 'i(', 'i)', 'ELSE', 'e(', 'e)']
    if type(node) == Conjunction:
        return []

    if type(node) == Not:
        return ['not', 'c(', 'c)']
    if type(node) == And:
        return ['and', 'c(', 'c)']
    if type(node) == Or:
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
    
    if token == '<HOLE>': return None

    return None

def _find_close_token(token_list: list[str], character: str, start_index: int = 0) -> int:
    open_token = character + '('
    close_token = character + ')'
    assert token_list[start_index] == open_token, 'Invalid program'
    parentheses = 1
    for i, t in enumerate(token_list[start_index+1:]):
        if t == open_token:
            parentheses += 1
        elif t == close_token:
            parentheses -= 1
        if parentheses == 0:
            return i + 1 + start_index
    raise Exception('Invalid program')

def _str_list_to_node(token_list: list[str]) -> Node:
    if len(token_list) == 0:
        return EmptyStatement()
    
    capitalized = token_list[0][0].upper() + token_list[0][1:]
    if capitalized in [c.__name__ for c in TerminalNode.__subclasses__()]:
        if len(token_list) > 1:
            s1 = globals()[capitalized]()
            s2 = _str_list_to_node(token_list[1:])
            return Conjunction.new(s1, s2)
        return globals()[capitalized]()
    
    if token_list[0] == '<HOLE>':
        if len(token_list) > 1:
            s1 = None
            s2 = _str_list_to_node(token_list[1:])
            return Conjunction.new(s1, s2)
        return None
    
    if token_list[0] == 'DEF':
        assert token_list[1] == 'run', 'Invalid program'
        assert token_list[2] == 'm(', 'Invalid program'
        assert token_list[-1] == 'm)', 'Invalid program'
        m = _str_list_to_node(token_list[3:-1])
        return Program.new(m)
    
    elif token_list[0] == 'IF':
        c_end = _find_close_token(token_list, 'c', 1)
        i_end = _find_close_token(token_list, 'i', c_end+1)
        c = _str_list_to_node(token_list[2:c_end])
        i = _str_list_to_node(token_list[c_end+2:i_end])
        if i_end == len(token_list) - 1: 
            return If.new(c, i)
        else:
            return Conjunction.new(
                If.new(c, i), 
                _str_list_to_node(token_list[i_end+1:])
            )
    elif token_list[0] == 'IFELSE':
        c_end = _find_close_token(token_list, 'c', 1)
        i_end = _find_close_token(token_list, 'i', c_end+1)
        assert token_list[i_end+1] == 'ELSE', 'Invalid program'
        e_end = _find_close_token(token_list, 'e', i_end+2)
        c = _str_list_to_node(token_list[2:c_end])
        i = _str_list_to_node(token_list[c_end+2:i_end])
        e = _str_list_to_node(token_list[i_end+3:e_end])
        if e_end == len(token_list) - 1: 
            return ITE.new(c, i, e)
        else:
            return Conjunction.new(
                ITE.new(c, i, e),
                _str_list_to_node(token_list[e_end+1:])
            )
    elif token_list[0] == 'WHILE':
        c_end = _find_close_token(token_list, 'c', 1)
        w_end = _find_close_token(token_list, 'w', c_end+1)
        c = _str_list_to_node(token_list[2:c_end])
        w = _str_list_to_node(token_list[c_end+2:w_end])
        if w_end == len(token_list) - 1: 
            return While.new(c, w)
        else:
            return Conjunction.new(
                While.new(c, w),
                _str_list_to_node(token_list[w_end+1:])
            )
    elif token_list[0] == 'REPEAT':
        n = _str_list_to_node([token_list[1]])
        r_end = _find_close_token(token_list, 'r', 2)
        r = _str_list_to_node(token_list[3:r_end])
        if r_end == len(token_list) - 1: 
            return Repeat.new(n, r)
        else:
            return Conjunction.new(
                Repeat.new(n, r),
                _str_list_to_node(token_list[r_end+1:])
            )
    
    elif token_list[0] == 'not':
        assert token_list[1] == 'c(', 'Invalid program'
        assert token_list[-1] == 'c)', 'Invalid program'
        c = _str_list_to_node(token_list[2:-1])
        return Not.new(c)
    elif token_list[0] == 'and':
        c1_end = _find_close_token(token_list, 'c', 1)
        assert token_list[c1_end+1] == 'c(', 'Invalid program'
        assert token_list[-1] == 'c)', 'Invalid program'
        c1 = _str_list_to_node(token_list[2:c1_end])
        c2 = _str_list_to_node(token_list[c1_end+2:-1])
        return And.new(c1, c2)
    elif token_list[0] == 'or':
        c1_end = _find_close_token(token_list, 'c', 1)
        assert token_list[c1_end+1] == 'c(', 'Invalid program'
        assert token_list[-1] == 'c)', 'Invalid program'
        c1 = _str_list_to_node(token_list[2:c1_end])
        c2 = _str_list_to_node(token_list[c1_end+2:-1])
        return Or.new(c1, c2)

    elif token_list[0].startswith('R='):
        num = int(token_list[0].replace('R=', ''))
        assert num is not None
        return ConstIntNode.new(num)
    else:
        raise Exception(f'Unrecognized token: {token_list[0]}.')

def _node_to_str(node: Node) -> str:
    if node is None:
        return '<HOLE>'
    
    if node.__class__ == ConstIntNode:
        return 'R=' + str(node.value)
    if node.__class__ == ConstBoolNode:
        return str(node.value)
    if node.__class__ in TerminalNode.__subclasses__():
        return node.name

    if node.__class__ == Program:
        m = _node_to_str(node.children[0])
        return f'DEF run m( {m} m)'

    if node.__class__ == While:
        c = _node_to_str(node.children[0])
        w = _node_to_str(node.children[1])
        return f'WHILE c( {c} c) w( {w} w)'
    if node.__class__ == Repeat:
        n = _node_to_str(node.children[0])
        r = _node_to_str(node.children[1])
        return f'REPEAT {n} r( {r} r)'
    if node.__class__ == If:
        c = _node_to_str(node.children[0])
        i = _node_to_str(node.children[1])
        return f'IF c( {c} c) i( {i} i)'
    if node.__class__ == ITE:
        c = _node_to_str(node.children[0])
        i = _node_to_str(node.children[1])
        e = _node_to_str(node.children[2])
        return f'IFELSE c( {c} c) i( {i} i) ELSE e( {e} e)'
    if node.__class__ == Conjunction:
        s1 = _node_to_str(node.children[0])
        s2 = _node_to_str(node.children[1])
        return f'{s1} {s2}'

    if node.__class__ == Not:
        c = _node_to_str(node.children[0])
        return f'not c( {c} c)'
    if node.__class__ == And:
        c1 = _node_to_str(node.children[0])
        c2 = _node_to_str(node.children[1])
        return f'and c( {c1} c) c( {c2} c)'
    if node.__class__ == Or:
        c1 = _node_to_str(node.children[0])
        c2 = _node_to_str(node.children[1])
        return f'or c( {c1} c) c( {c2} c)'


class DSL:

    def __init__(self, nodes: list[Node] = None, tokens: list[str] = None):
        self.nodes = nodes
        self.tokens = tokens
        self.t2i = {token: i for i, token in enumerate(self.tokens)}
        self.i2t = {i: token for i, token in enumerate(self.tokens)}
        self.actions = [
            n for n in self.nodes
            if n.__class__ in StatementNode.__subclasses__()
            and n.__class__ in TerminalNode.__subclasses__()
        ]

    @classmethod
    def init_from_nodes(cls, nodes):
        tokens = ['DEF', 'run', 'm(', 'm)']
        for node in nodes:
            tokens += _get_token_list_from_node(node)
        tokens = list(dict.fromkeys(tokens)) # Remove duplicates
        return cls(nodes, tokens)

    @classmethod
    def init_from_tokens(cls, tokens):
        nodes = [Conjunction()]
        for token in tokens:
            node = _get_node_from_token(token)
            if node is not None:
                nodes.append(node)
        return cls(nodes, tokens)

    @classmethod
    def init_default_karel(cls):
        nodes = [ConstIntNode.new(i) for i in range(20)]
        nodes += [While(), Repeat(), If(), ITE(), Conjunction(), Not(), FrontIsClear(),
                  RightIsClear(), LeftIsClear(), MarkersPresent(), NoMarkersPresent(),
                  Move(), TurnLeft(), TurnRight(), PickMarker(), PutMarker()]
        tokens = [
            'DEF', 'run', 'm(', 'm)', 'move', 'turnRight', 'turnLeft', 'pickMarker', 'putMarker',
            'r(', 'r)', 'R=0', 'R=1', 'R=2', 'R=3', 'R=4', 'R=5', 'R=6', 'R=7', 'R=8', 'R=9', 'R=10',
            'R=11', 'R=12', 'R=13', 'R=14', 'R=15', 'R=16', 'R=17', 'R=18', 'R=19', 'REPEAT', 'c(',
            'c)', 'i(', 'i)', 'e(', 'e)', 'IF', 'IFELSE', 'ELSE', 'frontIsClear', 'leftIsClear',
            'rightIsClear', 'markersPresent', 'noMarkersPresent', 'not', 'w(', 'w)', 'WHILE',
            '<pad>'
        ]
        if Config.dsl_include_hole:
            nodes.append(None)
            tokens.append('<HOLE>')
        return DSL(nodes, tokens)

    def get_actions(self) -> list[Node]:
        return self.actions

    def get_tokens(self) -> list[str]:
        return self.tokens
    
    def parse_node_to_int(self, node: Node) -> list[int]:
        prog_str = self.parse_node_to_str(node)
        return self.parse_str_to_int(prog_str)
    
    def parse_int_to_node(self, prog_tokens: list[int]) -> Node:
        prog_str = self.parse_int_to_str(prog_tokens)
        return self.parse_str_to_node(prog_str)
    
    def parse_node_to_str(self, node: Node) -> str:
        return _node_to_str(node)
    
    def parse_str_to_node(self, prog_str: str) -> Node:
        prog_str_list = prog_str.split(' ')
        return _str_list_to_node(prog_str_list)
    
    def parse_int_to_str(self, prog_tokens: list[int]) -> str:
        token_list = [self.i2t[i] for i in prog_tokens]
        return ' '.join(token_list)
    
    def parse_str_to_int(self, prog_str: str) -> list[int]:
        prog_str_list = prog_str.split(' ')
        return [self.t2i[i] for i in prog_str_list]
    
    def pad_tokens(self, prog_tokens: list[int], length: int) -> list[int]:
        return prog_tokens + [self.t2i['<pad>']] * (length - len(prog_tokens))