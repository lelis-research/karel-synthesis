import pickle
from PIL import Image
from dsl.parser import Parser
from karel.world import World
from karel.environment import Environment


if __name__ == '__main__':
    
    with open('leaps_results/LEAPS_CEM_MAZE-handwritten-2-20221216-005905/CEM/413_best_exec_data.pkl', 'rb') as f:
        info = pickle.load(f)
        
    prog = info[0]['exec_data']['program_prediction']
    
    s_0 = info[0]['exec_data']['s_h'][0][0]
    world = World(s_0)

    im = Image.fromarray(world.to_image())
    im_list = []
    for s in info[0]['exec_data']['s_h'][0]:
        # print(s.shape)
        try:
            w = World(s)
            im_list.append(Image.fromarray(w.to_image()))
        except IndexError:
            pass
        # print(world.to_string())
    im.save('output/leaps_states.gif', save_all=True, append_images=im_list, duration=75, loop=0)
    
    # world = World(s)
    # print(world.to_string())
    # for a in exec_data[0]['exec_data']['a_h'][0]:
    #     world.run_action(a)
    #     print(world.to_string())
    
    print(world.to_string())
    print(info[0]['exec_data']['a_h'][0])
    print(prog)
    
    program = Parser.str_to_nodes(prog)
    env = Environment(world, program)
    env.run_and_trace(f'output/leaps_result.gif')