from config.config import Config
from dsl.parser import Parser
from karel.world import World
from tasks.stair_climber import StairClimber
from tasks.task_environment import TaskEnvironment

PROG = 'DEF run m( WHILE c( NoMarkersPresent c) w( TurnLeft Move Move Move Move w) m)'


if __name__ == '__main__':
    
    Config.env_seed = 5
    
    task = StairClimber()
    
    env = TaskEnvironment(task)
    
    prog = Parser.tokens_to_nodes(PROG)
    
    states_history, reward = env.execute_agent(prog)
    
    for s in states_history:
        print(World(s).to_string())
        print()
    
    pass