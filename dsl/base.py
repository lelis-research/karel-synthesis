from typing import Union
from karel.world import World

class Node:

    def __init__(self):
        self.node_size: int = 1
        self.number_children: int = 0
        self.children: list[Node] = []

    def add_child(self, child: "Node"):            
        if len(self.children) + 1 > self.number_children:
            raise Exception('Unsupported number of children')
        self.children.append(child)

    def get_current_child(self) -> int:
        return len(self.children)
    
    def get_number_children(self) -> int:
        return self.number_children
    
    def get_size(self) -> int:
        size = self.node_size
        for child in self.children:
            size += child.get_size()
        return size

    @classmethod
    def get_children_types(cls):
        return []
        
    def replace_child(self, child: "Node", i: int) -> None:
        if len(self.children) < i + 1:
            self.add_child(child)
        else:
            self.children[i] = child
    
    def interpret(self, env: World) -> None:
        raise Exception('Unimplemented method: interpret')

    def is_complete(self):
        if self.number_children == 0:
            return True
        elif self.number_children == len(self.children):
            complete = True
            for child in self.children:
                if not child.is_complete():
                    complete = False
            return complete
        else:
            return False


# Node types, for inheritance to other classes
# Int: integer functions/constants (int return)
# Bool: boolean functions/constants (bool return)
# Statement: expression or terminal action functions (no return)
class IntNode(Node):

    def interpret(self, env: World) -> int:
        raise Exception('Unimplemented method: interpret')


class BoolNode(Node):

    def interpret(self, env: World) -> bool:
        raise Exception('Unimplemented method: interpret')


class StatementNode(Node):
    
    def interpret(self, env: World) -> None:
        raise Exception('Unimplemented method: interpret')


# Terminal/Non-Terminal types, for inheritance to other classes
class TerminalNode(Node): pass


class OperationNode(Node): pass


# Constants
class ConstBoolNode(BoolNode, TerminalNode):
    
    def __init__(self):
        super(ConstBoolNode, self).__init__()
        self.value: bool = False

    @classmethod
    def new(cls, value: bool):
        inst = cls()
        inst.value = value
        return inst

    def interpret(self, env: World) -> bool:
        return self.value


class ConstIntNode(IntNode, TerminalNode):
    
    def __init__(self):
        super(ConstIntNode, self).__init__()
        self.value: int = 0

    @classmethod
    def new(cls, value: int):
        inst = cls()
        inst.value = value
        return inst

    def interpret(self, env: World) -> int:
        return self.value


# Program as an abritrary statement
class Program(Node):

    def __init__(self):
        super(Program, self).__init__()
        self.number_children = 1
        self.node_size = 0
        self.children: list[StatementNode] = []

    @classmethod
    def get_children_types(cls):
        return [StatementNode]
        
    @classmethod
    def new(cls, var: StatementNode):
        inst = cls()
        inst.add_child(var)
        return inst
    
    def interpret(self, env: World):
        if len(self.children) == 0:
            raise Exception(f'{type(self).__name__}: Incomplete Program')
        return self.children[0].interpret(env)


# Expressions
class While(StatementNode, OperationNode):

    def __init__(self):
        super(While, self).__init__()
        self.number_children = 2
        self.children: list[Union[BoolNode, StatementNode]] = []

    @classmethod
    def new(cls, bool_expression: BoolNode, statement: StatementNode):
        inst = cls()
        inst.add_child(bool_expression)
        inst.add_child(statement)
        return inst

    @classmethod
    def get_children_types(cls):
        return [BoolNode, StatementNode]

    def interpret(self, env: World) -> None:
        while self.children[0].interpret(env):
            self.children[1].interpret(env)
            if env.crashed: break


class Repeat(StatementNode, OperationNode):

    def __init__(self):
        super(Repeat, self).__init__()
        self.number_children = 2
        self.children: list[Union[IntNode, StatementNode]] = []

    @classmethod
    def new(cls, number_repeats: IntNode, statement: StatementNode):
        inst = cls()
        inst.add_child(number_repeats)
        inst.add_child(statement)
        return inst

    @classmethod
    def get_children_types(cls):
        return [IntNode, StatementNode]

    def interpret(self, env: World) -> None:
        for _ in range(self.children[0].interpret(env)):
            self.children[1].interpret(env)


class If(StatementNode, OperationNode):

    def __init__(self):
        super(If, self).__init__()
        self.number_children = 2
        self.children: list[Union[BoolNode, StatementNode]] = []

    @classmethod
    def new(cls, bool_expression: BoolNode, statement: StatementNode):
        inst = cls()
        inst.add_child(bool_expression)
        inst.add_child(statement)
        return inst

    @classmethod
    def get_children_types(cls):
        return [BoolNode, StatementNode]

    def interpret(self, env: World) -> None:
        if self.children[0].interpret(env):
            self.children[1].interpret(env)


class ITE(StatementNode, OperationNode):

    def __init__(self):
        super(ITE, self).__init__()
        self.number_children = 3
        self.children: list[Union[BoolNode, StatementNode]] = []

    @classmethod
    def new(cls, bool_expression: BoolNode, statement_true: StatementNode, statement_false: StatementNode):
        inst = cls()
        inst.add_child(bool_expression)
        inst.add_child(statement_true)
        inst.add_child(statement_false)
        return inst

    @classmethod
    def get_children_types(cls):
        return [BoolNode, StatementNode, StatementNode]

    def interpret(self, env: World) -> None:
        if self.children[0].interpret(env):
            self.children[1].interpret(env)
        else:
            self.children[2].interpret(env)


class Conjunction(StatementNode, OperationNode):

    def __init__(self):
        super(Conjunction, self).__init__()
        self.number_children = 2
        self.children: list[StatementNode] = []

    @classmethod
    def new(cls, left_statement: StatementNode, right_statement: StatementNode):
        inst = cls()
        inst.add_child(left_statement)
        inst.add_child(right_statement)
        return inst

    @classmethod
    def get_children_types(cls):
        return [StatementNode, StatementNode]

    def interpret(self, env: World) -> None:
        self.children[0].interpret(env)
        self.children[1].interpret(env)


# Boolean functions
class FrontIsClear(BoolNode, TerminalNode):

    def interpret(self, env: World) -> bool:
        return env.front_is_clear()


class LeftIsClear(BoolNode, TerminalNode):

    def interpret(self, env: World) -> bool:
        return env.left_is_clear()


class RightIsClear(BoolNode, TerminalNode):

    def interpret(self, env: World) -> bool:
        return env.right_is_clear()


class MarkersPresent(BoolNode, TerminalNode):

    def interpret(self, env: World) -> bool:
        return env.markers_present()

# Note: the original implementation also declares NoMarkersPresent,
#       but as it can be created with Not(MarkersPresent), I did not
#       implement it in this code.


# Boolean operations
class Not(BoolNode, OperationNode):

    def __init__(self):
        super(Not, self).__init__()
        self.number_children = 1

    @classmethod
    def new(cls, var):
        inst = cls()
        inst.add_child(var)        
        return inst

    @classmethod
    def get_children_types(cls):
        return [BoolNode]
    
    def interpret(self, env: World) -> bool:
        if len(self.children) == 0:
            raise Exception(f'{type(self).__name__}: Incomplete Program')
        return not self.children[0].interpret(env)


# Note: And and Or are defined here but are not used in Karel
class And(BoolNode, OperationNode):

    def __init__(self):
        super(And, self).__init__()
        self.number_children = 2

    @classmethod
    def new(cls, left, right):
        inst = cls()
        inst.add_child(left)
        inst.add_child(right)
        return inst

    @classmethod
    def get_children_types(cls):
        return [BoolNode, BoolNode]
    
    def interpret(self, env: World) -> bool:
        if len(self.children) < 2:
            raise Exception(f'{type(self).__name__}: Incomplete Program')
        return self.children[0].interpret(env) and self.children[1].interpret(env)


class Or(BoolNode, OperationNode):

    def __init__(self):
        super(Or, self).__init__()
        self.number_children = 2

    @classmethod
    def new(cls, left, right):
        inst = cls()
        inst.add_child(left)
        inst.add_child(right)
        return inst

    @classmethod
    def get_children_types(cls):
        return [BoolNode, BoolNode]
    
    def interpret(self, env: World) -> bool:
        if len(self.children) < 2:
            raise Exception(f'{type(self).__name__}: Incomplete Program')
        return self.children[0].interpret(env) or self.children[1].interpret(env)


# Terminal actions
class Move(StatementNode, TerminalNode):

    def interpret(self, env: World) -> None:
        env.move()


class TurnLeft(StatementNode, TerminalNode):

    def interpret(self, env: World) -> None:
        env.turn_left()


class TurnRight(StatementNode, TerminalNode):

    def interpret(self, env: World) -> None:
        env.turn_right()


class PickMarker(StatementNode, TerminalNode):

    def interpret(self, env: World) -> None:
        env.pick_marker()


class PutMarker(StatementNode, TerminalNode):

    def interpret(self, env: World) -> None:
        env.put_marker()
