from __future__ import annotations
from config import Config
from dsl import DSL
from search.top_down import TopDownSearch
from tasks import get_task_cls

if __name__ == '__main__':
    
    Config.dsl_include_hole = True
    
    dsl = DSL.init_default_karel()
    
    task = get_task_cls(Config.env_task)
    
    incomplete_program = dsl.parse_str_to_node('DEF run m( WHILE c( noMarkersPresent c) w( turnLeft <HOLE> <HOLE> w) m)')
    
    filled_program, num_eval, converged = TopDownSearch().synthesize(incomplete_program, dsl, task, 4)

    print(dsl.parse_node_to_str(filled_program))
