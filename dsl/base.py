from typing import Generator, Union
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
    
    # In this implementation, get_size is run recursively in a program, so we do not need to worry
    # about updating each node size as we grow them
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
    
    # interpret is used by nodes that return a value (IntNode, BoolNode)
    def interpret(self, env: World) -> Union[bool, int]:
        raise Exception('Unimplemented method: interpret')

    # run and run_generator are used by nodes that affect env (StatementNode)
    def run(self, env: World) -> None:
        raise Exception('Unimplemented method: run')

    def run_generator(self, env: World) -> Generator[type, None, None]:
        raise Exception('Unimplemented method: run_generator')

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


class StatementNode(Node): pass


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

    def run(self, env: World) -> None:
        if len(self.children) == 0:
            raise Exception(f'{type(self).__name__}: Incomplete Program')
        self.children[0].run(env)
    
    def run_generator(self, env: World) -> Generator[type, None, None]:
        if len(self.children) == 0:
            raise Exception(f'{type(self).__name__}: Incomplete Program')
        yield from self.children[0].run_generator(env)


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

    def run(self, env: World) -> None:
        while self.children[0].interpret(env):
            if env.is_crashed(): return     # To avoid infinite loops
            self.children[1].run(env)

    def run_generator(self, env: World):
        while self.children[0].interpret(env):
            if env.is_crashed(): return     # To avoid infinite loops
            yield from self.children[1].run_generator(env)


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

    def run(self, env: World) -> None:
        for _ in range(self.children[0].interpret(env)):
            self.children[1].run(env)

    def run_generator(self, env: World):
        for _ in range(self.children[0].interpret(env)):
            yield from self.children[1].run_generator(env)


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

    def run(self, env: World) -> None:
        if self.children[0].interpret(env):
            self.children[1].run(env)

    def run_generator(self, env: World):
        if self.children[0].interpret(env):
            yield from self.children[1].run_generator(env)


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

    def run(self, env: World) -> None:
        if self.children[0].interpret(env):
            self.children[1].run(env)
        else:
            self.children[2].run(env)

    def run_generator(self, env: World):
        if self.children[0].interpret(env):
            yield from self.children[1].run_generator(env)
        else:
            yield from self.children[2].run_generator(env)


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

    def run(self, env: World) -> None:
        self.children[0].run(env)
        self.children[1].run(env)

    def run_generator(self, env: World):
        yield from self.children[0].run_generator(env)
        yield from self.children[1].run_generator(env)


# Terminal actions
class Move(StatementNode, TerminalNode):

    def run(self, env: World) -> None:
        env.move()

    def run_generator(self, env: World):
        env.move()
        yield Move


class TurnLeft(StatementNode, TerminalNode):

    def run(self, env: World) -> None:
        env.turn_left()

    def run_generator(self, env: World):
        env.turn_left()
        yield TurnLeft


class TurnRight(StatementNode, TerminalNode):

    def run(self, env: World) -> None:
        env.turn_right()

    def run_generator(self, env: World):
        env.turn_right()
        yield TurnRight


class PickMarker(StatementNode, TerminalNode):

    def run(self, env: World) -> None:
        env.pick_marker()

    def run_generator(self, env: World):
        env.pick_marker()
        yield PickMarker


class PutMarker(StatementNode, TerminalNode):

    def run(self, env: World) -> None:
        env.put_marker()

    def run_generator(self, env: World):
        env.put_marker()
        yield PutMarker


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


class NoMarkersPresent(BoolNode, TerminalNode):

    def interpret(self, env: World) -> bool:
        return not env.markers_present()