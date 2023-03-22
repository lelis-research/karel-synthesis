from karel.data import Data
from dsl import DSL
from dsl.parser import Parser
from dsl.base import *
from karel.environment import Environment
from search.bottom_up import BottomUpSearch

# TODO: rewrite this using rewards from program execution (similar to top-down)
if __name__ == '__main__':

    data = Data.from_json('data/1m_6ex_karel/train.json', 1)
    prod = DSL.init_default_karel()

    synthetizer = BottomUpSearch()

    program, num_eval = synthetizer.synthesize(data, prod, 10)

    print(Parser.nodes_to_str(program))

    for i, inp in enumerate(data.inputs):
        env = Environment(inp, program)
        env.run_and_trace(f'output/bus_{i}.gif')
