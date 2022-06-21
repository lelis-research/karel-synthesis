import copy
from dsl.base import Program
from karel.world import World

MAX_API_CALLS = 10000

# Environment encapsulates an agent (Program) and a Karel map (World) to allow simpler calls for
# executions. Please check the sample programs for usage examples.
class Environment:
    
    def __init__(self, initial_state: World, agent: Program) -> None:
        self.initial_state = initial_state
        self.agent = agent
        self.restart_agent()

    def replace_agent(self, new_agent: Program):
        self.agent = new_agent
        self.restart_agent()

    def restart_agent(self):
        self.state = copy.deepcopy(self.initial_state)
        if self.agent is not None:
            self.agent_generator = self.agent.run_generator(self.state)
    
    def run_agent(self):
        self.agent.run(self.state)

    def run_and_get_actions(self):
        return list(self.agent_generator)

    def run_single_action(self):
        return next(self.agent_generator)

    def get_world_state(self):
        return self.state

    def set_world_state(self, new_state: World):
        self.initial_state = new_state
        self.restart_agent()

    def run_and_trace(self, image_name = 'trace.gif'):
        from PIL import Image
        im = Image.fromarray(self.state.to_image())
        im_list = []
        for _ in self.agent_generator:
            im_list.append(Image.fromarray(self.state.to_image()))
        im.save(image_name, save_all=True, append_images=im_list, duration=75, loop=0)