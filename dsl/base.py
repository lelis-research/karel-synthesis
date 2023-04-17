from __future__ import annotations
from typing import Generator, Union
from karel.world import World

class Node:

    node_size: int = 1
    node_depth: int = 0
    children_types: list[type[Node]] = []

    def __init__(self, name: Union[str, None] = None):
        self.children: list[Union[Node, None]] = [None for _ in range(self.get_number_children())]
        self.value: Union[None, bool, int] = None
        if name is not None:
            self.name = name
        else:
            self.name = type(self).__name__
    
    # In this implementation, get_size is run recursively in a program, so we do not need to worry
    # about updating each node size as we grow them
    def get_size(self) -> int:
        size = self.node_size
        for child in self.children:
            if child is not None:
                size += child.get_size()
        return size
    
    # recursively calculate the node depth (number of levels from root)
    def get_depth(self) -> int:
        depth = 0
        for child in self.children:
            if child is not None:
                depth = max(depth, child.get_depth())
        return depth + self.node_depth
    
    # Recursively get all nodes in the tree
    def get_all_nodes(self) -> list[Node]:
        nodes = [self]
        for child in self.children:
            if child is not None:
                nodes += child.get_all_nodes()
        return nodes
    
    @classmethod
    def get_number_children(cls) -> int:
        return len(cls.children_types)

    @classmethod
    def get_children_types(cls) -> list[type[Node]]:
        return cls.children_types
    
    @classmethod
    def get_node_size(cls) -> int:
        return cls.node_size
    
    @classmethod
    def get_node_depth(cls) -> int:
        return cls.node_depth
    
    @classmethod
    def new(cls, *args) -> Node:
        inst = cls()
        children_types = cls.get_children_types()
        for i, arg in enumerate(args):
            if arg is not None:
                assert issubclass(type(arg), children_types[i])
                inst.children[i] = arg
        return inst
    
    # interpret is used by nodes that return a value (IntNode, BoolNode)
    def interpret(self, env: World) -> Union[bool, int]:
        raise Exception('Unimplemented method: interpret')

    # run and run_generator are used by nodes that affect env (StatementNode)
    def run(self, env: World) -> None:
        raise Exception('Unimplemented method: run')

    def run_generator(self, env: World) -> Generator[type, None, None]:
        raise Exception('Unimplemented method: run_generator')

    def is_complete(self) -> bool:
        for child in self.children:
            if child is None:
                return False
            elif not child.is_complete():
                return False
        return True


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
        super().__init__()
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
        super().__init__()
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

    node_size = 0
    node_depth = 1
    children_types = [StatementNode]

    def run(self, env: World) -> None:
        assert self.is_complete(), 'Incomplete Program'
        self.children[0].run(env)
    
    def run_generator(self, env: World) -> Generator[type, None, None]:
        assert self.is_complete(), 'Incomplete Program'
        yield from self.children[0].run_generator(env)


# Expressions
class While(StatementNode, OperationNode):

    node_depth = 1
    children_types = [BoolNode, StatementNode]

    def run(self, env: World) -> None:
        while self.children[0].interpret(env):
            if env.is_crashed(): return     # To avoid infinite loops
            self.children[1].run(env)

    def run_generator(self, env: World):
        while self.children[0].interpret(env):
            if env.is_crashed(): return     # To avoid infinite loops
            yield from self.children[1].run_generator(env)


class Repeat(StatementNode, OperationNode):

    node_depth = 1
    children_types = [IntNode, StatementNode]

    def run(self, env: World) -> None:
        for _ in range(self.children[0].interpret(env)):
            self.children[1].run(env)

    def run_generator(self, env: World):
        for _ in range(self.children[0].interpret(env)):
            yield from self.children[1].run_generator(env)


class If(StatementNode, OperationNode):

    node_depth = 1
    children_types = [BoolNode, StatementNode]

    def run(self, env: World) -> None:
        if self.children[0].interpret(env):
            self.children[1].run(env)

    def run_generator(self, env: World):
        if self.children[0].interpret(env):
            yield from self.children[1].run_generator(env)


class ITE(StatementNode, OperationNode):

    node_depth = 1
    children_types = [BoolNode, StatementNode, StatementNode]

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

    node_size = 0
    children_types = [StatementNode, StatementNode]

    def run(self, env: World) -> None:
        self.children[0].run(env)
        self.children[1].run(env)

    def run_generator(self, env: World):
        yield from self.children[0].run_generator(env)
        yield from self.children[1].run_generator(env)


# EmptyStatement sometimes shows up during VAE decoding, but
# preferably it should not be used in the final program
class EmptyStatement(StatementNode, TerminalNode):
    
    node_size = 0

    def __init__(self):
        super().__init__('empty')

    def run(self, env: World) -> None:
        return

    def run_generator(self, env: World):
        yield None # TODO: maybe replace by pass?
        # I did not worry about this because I want to eventually
        # remove this class, once SyntaxChecker does not allow
        # empty statements to be generated


# Terminal actions
class Move(StatementNode, TerminalNode):

    def __init__(self):
        super().__init__('move')

    def run(self, env: World) -> None:
        env.move()

    def run_generator(self, env: World):
        env.move()
        yield self


class TurnLeft(StatementNode, TerminalNode):

    def __init__(self):
        super().__init__('turnLeft')

    def run(self, env: World) -> None:
        env.turn_left()

    def run_generator(self, env: World):
        env.turn_left()
        yield self


class TurnRight(StatementNode, TerminalNode):

    def __init__(self):
        super().__init__('turnRight')

    def run(self, env: World) -> None:
        env.turn_right()

    def run_generator(self, env: World):
        env.turn_right()
        yield self


class PickMarker(StatementNode, TerminalNode):

    def __init__(self):
        super().__init__('pickMarker')

    def run(self, env: World) -> None:
        env.pick_marker()

    def run_generator(self, env: World):
        env.pick_marker()
        yield self


class PutMarker(StatementNode, TerminalNode):

    def __init__(self):
        super().__init__('putMarker')

    def run(self, env: World) -> None:
        env.put_marker()

    def run_generator(self, env: World):
        env.put_marker()
        yield self


# Boolean operations
class Not(BoolNode, OperationNode):

    children_types = [BoolNode]
    
    def interpret(self, env: World) -> bool:
        return not self.children[0].interpret(env)


# Note: And and Or are defined here but are not used in Karel
class And(BoolNode, OperationNode):

    children_types = [BoolNode, BoolNode]
    
    def interpret(self, env: World) -> bool:
        return self.children[0].interpret(env) and self.children[1].interpret(env)


class Or(BoolNode, OperationNode):

    children_types = [BoolNode, BoolNode]
    
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