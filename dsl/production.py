from .dsl import *

class Production:
    
    # Defines the production rules for the children of a node
    production_rules = {
        Program: [
            [While, Repeat, If, ITE, Conjunction, Move, TurnLeft, TurnRight, PutMarker, PickMarker]
        ],
        While: [
            [Not, FrontIsClear, LeftIsClear, RightIsClear, MarkersPresent, NoMarkersPresent],
            [While, Repeat, If, ITE, Conjunction, Move, TurnLeft, TurnRight, PutMarker, PickMarker]
        ],
        Repeat: [
            [ConstIntNode],
            [While, Repeat, If, ITE, Conjunction, Move, TurnLeft, TurnRight, PutMarker, PickMarker]
        ],
        If: [
            [Not, FrontIsClear, LeftIsClear, RightIsClear, MarkersPresent, NoMarkersPresent],
            [While, Repeat, If, ITE, Conjunction, Move, TurnLeft, TurnRight, PutMarker, PickMarker]
        ],
        ITE: [
            [FrontIsClear, LeftIsClear, RightIsClear, MarkersPresent, NoMarkersPresent],
            [While, Repeat, If, ITE, Conjunction, Move, TurnLeft, TurnRight, PutMarker, PickMarker],
            [While, Repeat, If, ITE, Conjunction, Move, TurnLeft, TurnRight, PutMarker, PickMarker]
        ],
        Conjunction: [
            [While, Repeat, If, ITE, Move, TurnLeft, TurnRight, PutMarker, PickMarker],
            [While, Repeat, If, ITE, Conjunction, Move, TurnLeft, TurnRight, PutMarker, PickMarker]
        ],
        Not: [
            [FrontIsClear, LeftIsClear, RightIsClear, MarkersPresent, NoMarkersPresent]
        ]
    }
    
    @staticmethod
    def get_production_rules(node_type: type[Node]) -> list[list[type[Node]]]:
        return Production.production_rules[node_type]