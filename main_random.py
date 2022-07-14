from karel.environment import Environment
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

    print('Program:', Parser.nodes_to_tokens(program))
    print('Program size:', program.get_size())

    for i, w in enumerate(worlds):

        world = World.from_string(w)

        env = Environment(world, program)
        env.run_and_trace(f'output/random_{i}.gif')
        env = Environment(world, program)
        env.run_and_trace(f'output/random_{i}.gif')
