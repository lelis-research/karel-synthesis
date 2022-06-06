from karel.world import World
from dsl.base import *
from dsl.parser import Parser

if __name__ == '__main__':

    program = Program.new(
        Conjunction.new(
            If.new(RightIsClear(), Conjunction.new(
                TurnRight(), Conjunction.new(
                    Move(), Conjunction.new(
                        PutMarker(), Conjunction.new(
                            TurnLeft(), Conjunction.new(
                                TurnLeft(), Conjunction.new(
                                    Move(), TurnRight()
                                )
                            )
                        )
                    )
                )
            )),
            While.new(FrontIsClear(), Conjunction.new(
                Move(), If.new(RightIsClear(), Conjunction.new(
                    TurnRight(), Conjunction.new(
                        Move(), Conjunction.new(
                            PutMarker(), Conjunction.new(
                                TurnLeft(), Conjunction.new(
                                    TurnLeft(), Conjunction.new(
                                        Move(), TurnRight()
                                    )
                                )
                            )
                        )
                    )
                ))
            ))
        )
    )

    worlds = [
        '|  |\n' +
        '|  |\n' +
        '|^*|',

        '|  |\n' +
        '| *|\n' +
        '|  |\n' +
        '|  |\n' +
        '| *|\n' +
        '|  |\n' +
        '|  |\n' +
        '| *|\n' +
        '|  |\n' +
        '|^ |',

        '|  |\n' +
        '| *|\n' +
        '| *|\n' +
        '| *|\n' +
        '|  |\n' +
        '|  |\n' +
        '| *|\n' +
        '| *|\n' +
        '| *|\n' +
        '|^ |'
    ]

    print('Program:', Parser.to_string(program))
    print('Program size:', program.get_size())

    for i, w in enumerate(worlds):
        print(f'World {i+1}:')

        world = World.from_string(w)

        print('Starting map:')
        print(world.to_string())

        program.interpret(world)

        print('Ending map:')
        print(world.to_string())
        print()
