from __future__ import annotations
from typing import Generator, Union
from karel.world import World

class Node:

    def __init__(self, name: Union[str, None] = None):
        self.node_size: int = 1
        self.number_children: int = 0
        self.children: list[Node] = []
        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__

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
            if child is not None:
                size += child.get_size()
        return size

    @classmethod
    def get_children_types(cls) -> list[type["Node"]]:
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
                if child is None:
                    complete = False
                elif not child.is_complete():
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


# Program as an arbitrary node with a single StatementNode child
class Program(Node):

    def __init__(self):
        super(Program, self).__init__()
        self.number_children = 1
        self.node_size = 0
        self.children: list[StatementNode] = [None]

    @classmethod
    def get_children_types(cls):
        return [StatementNode]
        
    @classmethod
    def new(cls, var: StatementNode):
        inst = cls()
        inst.replace_child(var, 0)
        return inst

    def run(self, env: World) -> None:
        assert self.is_complete(), 'Incomplete Program'
        self.children[0].run(env)
    
    def run_generator(self, env: World) -> Generator[type, None, None]:
        assert self.is_complete(), 'Incomplete Program'
        yield from self.children[0].run_generator(env)


# Expressions
class While(StatementNode, OperationNode):

    def __init__(self):
        super(While, self).__init__()
        self.number_children = 2
        self.children: list[Union[BoolNode, StatementNode]] = [None, None]

    @classmethod
    def new(cls, bool_expression: BoolNode, statement: StatementNode):
        inst = cls()
        inst.replace_child(bool_expression, 0)
        inst.replace_child(statement, 1)
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
        self.children: list[Union[IntNode, StatementNode]] = [None, None]

    @classmethod
    def new(cls, number_repeats: IntNode, statement: StatementNode):
        inst = cls()
        inst.replace_child(number_repeats, 0)
        inst.replace_child(statement, 1)
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
        self.children: list[Union[BoolNode, StatementNode]] = [None, None]

    @classmethod
    def new(cls, bool_expression: BoolNode, statement: StatementNode):
        inst = cls()
        inst.replace_child(bool_expression, 0)
        inst.replace_child(statement, 1)
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
        self.children: list[Union[BoolNode, StatementNode]] = [None, None, None]

    @classmethod
    def new(cls, bool_expression: BoolNode, statement_true: StatementNode, statement_false: StatementNode):
        inst = cls()
        inst.replace_child(bool_expression, 0)
        inst.replace_child(statement_true, 1)
        inst.replace_child(statement_false, 2)
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
        self.children: list[StatementNode] = [None, None]

    @classmethod
    def new(cls, left_statement: StatementNode, right_statement: StatementNode):
        inst = cls()
        inst.replace_child(left_statement, 0)
        inst.replace_child(right_statement, 1)
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

    def __init__(self):
        super().__init__('move')

    def run(self, env: World) -> None:
        env.move()

    def run_generator(self, env: World):
        env.move()
        yield Move


class EmptyStatement(StatementNode, TerminalNode):
    
    def __init__(self):
        super().__init__('empty')
        self.node_size = 0

    def run(self, env: World) -> None:
        return

    def run_generator(self, env: World):
        return


class TurnLeft(StatementNode, TerminalNode):

    def __init__(self):
        super().__init__('turnLeft')

    def run(self, env: World) -> None:
        env.turn_left()

    def run_generator(self, env: World):
        env.turn_left()
        yield TurnLeft


class TurnRight(StatementNode, TerminalNode):

    def __init__(self):
        super().__init__('turnRight')

    def run(self, env: World) -> None:
        env.turn_right()

    def run_generator(self, env: World):
        env.turn_right()
        yield TurnRight


class PickMarker(StatementNode, TerminalNode):

    def __init__(self):
        super().__init__('pickMarker')

    def run(self, env: World) -> None:
        env.pick_marker()

    def run_generator(self, env: World):
        env.pick_marker()
        yield PickMarker


class PutMarker(StatementNode, TerminalNode):

    def __init__(self):
        super().__init__('putMarker')

    def run(self, env: World) -> None:
        env.put_marker()

    def run_generator(self, env: World):
        env.put_marker()
        yield PutMarker


# Boolean operations
class Not(BoolNode, OperationNode):

    def __init__(self):
        super().__init__()
        self.children: list[BoolNode] = [None]
        self.number_children = 1

    @classmethod
    def new(cls, var):
        inst = cls()
        inst.replace_child(var, 0)        
        return inst

    @classmethod
    def get_children_types(cls):
        return [BoolNode]
    
    def interpret(self, env: World) -> bool:
        return not self.children[0].interpret(env)


# Note: And and Or are defined here but are not used in Karel
class And(BoolNode, OperationNode):

    def __init__(self):
        super().__init__()
        self.children: list[BoolNode] = [None, None]
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
        return self.children[0].interpret(env) and self.children[1].interpret(env)


class Or(BoolNode, OperationNode):

    def __init__(self):
        super().__init__()
        self.children: list[BoolNode] = [None, None]
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
        return self.children[0].interpret(env) or self.children[1].interpret(env)


# Boolean functions
class FrontIsClear(BoolNode, TerminalNode):

    def __init__(self):
        super().__init__('frontIsClear')

    def interpret(self, env: World) -> bool:
        return env.front_is_clear()


class LeftIsClear(BoolNode, TerminalNode):

    def __init__(self):
        super().__init__('leftIsClear')

    def interpret(self, env: World) -> bool:
        return env.left_is_clear()


class RightIsClear(BoolNode, TerminalNode):

    def __init__(self):
        super().__init__('rightIsClear')

    def interpret(self, env: World) -> bool:
        return env.right_is_clear()


class MarkersPresent(BoolNode, TerminalNode):

    def __init__(self):
        super().__init__('markersPresent')

    def interpret(self, env: World) -> bool:
        return env.markers_present()


class NoMarkersPresent(BoolNode, TerminalNode):

    def __init__(self):
        super().__init__('noMarkersPresent')

    def interpret(self, env: World) -> bool:
        return not env.markers_present()