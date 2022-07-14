from .base import *

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

def _tokens_to_node(token_list: list[str]) -> Node:
    if token_list[0] in [c.__name__ for c in TerminalNode.__subclasses__()]:
        if len(token_list) > 1:
            s1 = globals()[token_list[0]]()
            s2 = _tokens_to_node(token_list[1:])
            return Conjunction.new(s1, s2)
        return globals()[token_list[0]]()
    
    if token_list[0] == 'DEF':
        assert token_list[1] == 'run', 'Invalid program'
        assert token_list[2] == 'm(', 'Invalid program'
        assert token_list[-1] == 'm)', 'Invalid program'
        m = _tokens_to_node(token_list[3:-1])
        return Program.new(m)
    
    elif token_list[0] == 'IF':
        c_end = _find_close_token(token_list, 'c', 1)
        i_end = _find_close_token(token_list, 'i', c_end+1)
        c = _tokens_to_node(token_list[2:c_end])
        i = _tokens_to_node(token_list[c_end+2:i_end])
        if i_end == len(token_list) - 1: 
            return If.new(c, i)
        else:
            return Conjunction.new(
                If.new(c, i), 
                _tokens_to_node(token_list[i_end+1:])
            )
    elif token_list[0] == 'IFELSE':
        c_end = _find_close_token(token_list, 'c', 1)
        i_end = _find_close_token(token_list, 'i', c_end+1)
        assert token_list[i_end+1] == 'ELSE', 'Invalid program'
        e_end = _find_close_token(token_list, 'e', i_end+2)
        c = _tokens_to_node(token_list[2:c_end])
        i = _tokens_to_node(token_list[c_end+2:i_end])
        e = _tokens_to_node(token_list[i_end+3:e_end])
        if e_end == len(token_list) - 1: 
            return ITE.new(c, i, e)
        else:
            return Conjunction.new(
                ITE.new(c, i, e),
                _tokens_to_node(token_list[e_end+1:])
            )
    elif token_list[0] == 'WHILE':
        c_end = _find_close_token(token_list, 'c', 1)
        w_end = _find_close_token(token_list, 'w', c_end+1)
        c = _tokens_to_node(token_list[2:c_end])
        w = _tokens_to_node(token_list[c_end+2:w_end])
        if w_end == len(token_list) - 1: 
            return While.new(c, w)
        else:
            return Conjunction.new(
                While.new(c, w),
                _tokens_to_node(token_list[w_end+1:])
            )
    elif token_list[0] == 'REPEAT':
        n = ConstIntNode.new(int(token_list[1].replace('R=', '')))
        r_end = _find_close_token(token_list, 'r', 2)
        r = _tokens_to_node(token_list[3:r_end])
        if r_end == len(token_list) - 1: 
            return Repeat.new(n, r)
        else:
            return Conjunction.new(
                Repeat.new(n, r),
                _tokens_to_node(token_list[r_end+1:])
            )
    
    elif token_list[0] == 'not':
        assert token_list[1] == 'c(', 'Invalid program'
        assert token_list[-1] == 'c)', 'Invalid program'
        c = _tokens_to_node(token_list[2:-1])
        return Not.new(c)
    elif token_list[0] == 'and':
        c1_end = _find_close_token(token_list, 'c', 1)
        assert token_list[c1_end+1] == 'c(', 'Invalid program'
        assert token_list[-1] == 'c)', 'Invalid program'
        c1 = _tokens_to_node(token_list[2:c1_end])
        c2 = _tokens_to_node(token_list[c1_end+2:-1])
        return And.new(c1, c2)
    elif token_list[0] == 'or':
        c1_end = _find_close_token(token_list, 'c', 1)
        assert token_list[c1_end+1] == 'c(', 'Invalid program'
        assert token_list[-1] == 'c)', 'Invalid program'
        c1 = _tokens_to_node(token_list[2:c1_end])
        c2 = _tokens_to_node(token_list[c1_end+2:-1])
        return Or.new(c1, c2)
    else:
        raise Exception('Unrecognized token.')

def _node_to_string(node: Node) -> str:
    if node.__class__ == ConstIntNode or node.__class__ == ConstBoolNode:
        return str(node.value)
    if node.__class__ in TerminalNode.__subclasses__():
        return node.__class__.__name__

    if node.__class__ == Program:
        m = _node_to_string(node.children[0])
        return f'DEF run m( {m} m)'

    if node.__class__ == While:
        c = _node_to_string(node.children[0])
        w = _node_to_string(node.children[1])
        return f'WHILE c( {c} c) w( {w} w)'
    if node.__class__ == Repeat:
        n = _node_to_string(node.children[0])
        r = _node_to_string(node.children[1])
        return f'REPEAT {n} r( {r} r)'
    if node.__class__ == If:
        c = _node_to_string(node.children[0])
        i = _node_to_string(node.children[1])
        return f'IF c( {c} c) i( {i} i)'
    if node.__class__ == ITE:
        c = _node_to_string(node.children[0])
        i = _node_to_string(node.children[1])
        e = _node_to_string(node.children[2])
        return f'IFELSE c( {c} c) i( {i} i) ELSE e( {e} e)'
    if node.__class__ == Conjunction:
        s1 = _node_to_string(node.children[0])
        s2 = _node_to_string(node.children[1])
        return f'{s1} {s2}'

    if node.__class__ == Not:
        c = _node_to_string(node.children[0])
        return f'not c( {c} c)'
    if node.__class__ == And:
        c1 = _node_to_string(node.children[0])
        c2 = _node_to_string(node.children[1])
        return f'and c( {c1} c) c( {c2} c)'
    if node.__class__ == Or:
        c1 = _node_to_string(node.children[0])
        c2 = _node_to_string(node.children[1])
        return f'or c( {c1} c) c( {c2} c)'


class Parser:

    TOKENS = [
        'DEF', 'run', 'm(', 'm)', 'move', 'turnRight', 'turnLeft', 'pickMarker', 'putMarker',
        'r(', 'r)', 'R=0', 'R=1', 'R=2', 'R=3', 'R=4', 'R=5', 'R=6', 'R=7', 'R=8', 'R=9', 'R=10',
        'R=11', 'R=12', 'R=13', 'R=14', 'R=15', 'R=16', 'R=17', 'R=18', 'R=19', 'REPEAT', 'c(',
        'c)', 'i(', 'i)', 'e(', 'e)', 'IF', 'IFELSE', 'ELSE', 'frontIsClear', 'leftIsClear',
        'rightIsClear', 'markersPresent', 'noMarkersPresent', 'not', 'w(', 'w)', 'WHILE', '<pad>'
    ]

    T2I = {token: i for i, token in enumerate(TOKENS)}
    I2T = {i: token for i, token in enumerate(TOKENS)}

    @staticmethod
    def nodes_to_tokens(node: Node) -> str:
        assert node.is_complete(), 'Incomplete program'
        return _node_to_string(node)

    @staticmethod
    def tokens_to_nodes(prog_str: str) -> Node:
        token_list = prog_str.split(' ')
        return _tokens_to_node(token_list)

    @staticmethod
    def tokens_to_list(prog_str: str) -> list[int]:
        token_list = prog_str.split(' ')
        return [Parser.T2I[i] for i in token_list]

    @staticmethod
    def pad_list(prog_list: list[int], length: int) -> list[int]:
        return prog_list + [Parser.T2I['<pad>'] for _ in range(length - len(prog_list))]

    @staticmethod
    def list_to_tokens(prog_list: list[int]) -> str:
        token_list = [Parser.I2T[i] for i in prog_list]
        return ' '.join(token_list)
        