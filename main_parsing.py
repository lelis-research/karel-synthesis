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
        print(f'World {i+1}:')

        world = World.from_string(w)

        print('Starting map:')
        print(world.to_string())

        program.interpret(world)

        print('Ending map:')
        print(world.to_string())
        print()
