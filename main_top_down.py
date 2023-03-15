from __future__ import annotations
from dsl.production import Production
from dsl.parser import Parser
from search.top_down import TopDownSearch
from tasks.stair_climber import StairClimber

if __name__ == '__main__':
    
    dsl = Production.default_karel_production()
    
    task = StairClimber()
    
    incomplete_program = Parser.str_to_nodes('DEF run m( WHILE c( noMarkersPresent c) w( turnLeft move <HOLE> <HOLE> w) m)')
    
    filled_program, num_eval, converged = TopDownSearch().synthesize(incomplete_program, dsl, task, 2)

    print(Parser.nodes_to_str(filled_program))
