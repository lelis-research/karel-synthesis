from karel.environment import Environment
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

    print('Program:', Parser.nodes_to_tokens(program))
    print('Program size:', program.get_size())

    for i, w in enumerate(worlds):

        world = World.from_string(w)

        env = Environment(world, program)
        env.run_and_trace(f'output/symbolic_{i}.gif')

        f_world = env.get_world_state()

        print(f_world.to_string())
