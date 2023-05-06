
from config.config import Config
from dsl.dsl import DSL
from karel.world import World
from tasks import get_task_cls


if __name__ == '__main__':
    
    Config.env_height = 8
    Config.env_width = 8
    Config.env_enable_leaps_behaviour = True
    
    task = get_task_cls("DoorKey")(1)
    print(task.initial_state.to_string())
    
    dsl = DSL.init_default_karel()
    program = dsl.parse_str_to_node('DEF run m( REPEAT R=11 r( turnRight pickMarker move IFELSE c( leftIsClear c) i( pickMarker move i) ELSE e( turnRight move e) r) m)')
    
    reward = task.evaluate_program(program)
    
    print(reward)
    print(task.get_state().to_string())
    
    task.trace_program(program, 'trace.gif', max_steps=200)
    