from __future__ import annotations
from dsl.production import Production
from dsl.parser import Parser
from search.sketch_sampler import SketchSampler
from search.top_down import TopDownSearch
from tasks.stair_climber import StairClimber


if __name__ == '__main__':
    
    complete_program = Parser.str_to_nodes('DEF run m( WHILE c( noMarkersPresent c) w( turnLeft move turnRight move w) m)')
    
    dsl = Production.default_karel_production()
    
    task = StairClimber
    
    sketch = SketchSampler().sample_sketch(complete_program, 3)
    
    print('Sketch:', Parser.nodes_to_str(sketch))

    filled_program, num_eval, converged = TopDownSearch().synthesize(sketch, dsl, task, 3)

    print('Reconstructed program:', Parser.nodes_to_str(filled_program))
