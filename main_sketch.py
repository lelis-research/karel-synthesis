from __future__ import annotations
from dsl import DSL
from dsl.parser import Parser
from search.sketch_sampler import SketchSampler
from search.top_down import TopDownSearch
from tasks.stair_climber import StairClimber


if __name__ == '__main__':
    
    dsl = DSL.init_default_karel()

    complete_program = dsl.parse_str_to_node('DEF run m( WHILE c( noMarkersPresent c) w( turnLeft move turnRight move w) m)')
    
    task = StairClimber
    
    sketch = SketchSampler().sample_sketch(complete_program, 3)
    
    print('Sketch:', dsl.parse_node_to_str(sketch))

    filled_program, num_eval, converged = TopDownSearch().synthesize(sketch, dsl, task, 3)

    print('Reconstructed program:', dsl.parse_node_to_str(filled_program))
