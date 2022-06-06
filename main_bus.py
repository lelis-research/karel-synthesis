from karel.data import Data
from dsl.production import Production
from dsl.parser import Parser
from dsl.base import *
from search.bottom_up import BottomUpSearch

if __name__ == '__main__':

    data = Data.from_json('data/1m_6ex_karel/train.json', 1)
    prod = Production.default_karel_production()

    synthetizer = BottomUpSearch()

    program, num_eval = synthetizer.synthesize(data, prod, 10)

    print(Parser.to_string(program))
