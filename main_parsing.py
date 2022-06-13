from karel.environment import Environment
from karel.world import World
from dsl.base import *
from dsl.parser import Parser

if __name__ == '__main__':

    program = Parser.from_string('DEF run m( IF c( RightIsClear c) i( TurnRight Move PutMarker\
 REPEAT R=2 r( TurnLeft r) Move TurnRight i) WHILE c( FrontIsClear c) w( Move IF c( RightIsClear c)\
 i( TurnRight Move PutMarker REPEAT R=2 r( TurnLeft r) Move TurnRight i) w) m)')

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

        world = World.from_string(w)

        env = Environment(world, program)
        env.run_and_trace(f'output/parsing_{i}.gif')
        env = Environment(world, program)
        env.run_and_trace(f'output/parsing_{i}.gif')
