from karel.world import World
from dsl.production import Production
from dsl.parser import Parser

if __name__ == '__main__':

    prod = Production.default_karel_production()

    print(f'Terminals: {[obj.__class__.__name__ for obj in prod.terminals]}')
    print(f'Operations: {[obj.__class__.__name__ for obj in prod.operations]}')

    program = prod.random_program(3)

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
