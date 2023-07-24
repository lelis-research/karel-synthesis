import numpy as np
import torch
from config import Config
from dsl import DSL
from vae.models import load_model
from tasks import get_task_cls


if __name__ == '__main__':
    
    LR = 0.05
    N_ROLLOUTS = 10
    EPS = 0.1
    
    dsl = DSL.init_default_karel()

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(Config.model_name, dsl, device)
    
    task_cls = get_task_cls(Config.env_task)
    task_env = task_cls(Config.env_seed)
    
    params = torch.load(Config.model_params_path, map_location=device)
    model.load_state_dict(params, strict=False)
    
    for param in model.parameters():
        param.requires_grad = False
    
    z = torch.randn(Config.model_hidden_size).to(device)
    z.requires_grad_(True)
    
    optim = torch.optim.Adam([z], lr=LR)
    
    counter_rollouts = 0
    cumreturn = 0.

    while True:
        
        task_env.initial_state = task_env.generate_state()
        task_env.reset_state()
        state_np = task_env.get_state().get_state()
        state_np = np.moveaxis(state_np,[-1,-2,-3], [-3,-1,-2])
        
        state_torch = torch.tensor(np.array([state_np]), dtype=torch.float32, device=device).unsqueeze(0)

        output = model.policy_executor_reward(z, task_env)

        log_probs, rewards = output
        
        log_probs = torch.stack(log_probs)
        
        rewards_to_go = torch.cumsum(torch.tensor(list(reversed(rewards)), dtype=torch.float32), dim=0)
        rewards_to_go = torch.flip(rewards_to_go, [0])
        
        cumreturn += rewards_to_go[-1].item()
        counter_rollouts += 1
    
        for r, l in zip(rewards_to_go, log_probs):
            obj = -r * l
            optim.zero_grad()
            obj.backward(retain_graph=True)
            optim.step()
    
        print(f'rollout {counter_rollouts}: {rewards_to_go[-1].item()}')
    
        if counter_rollouts == N_ROLLOUTS:
            avgreturn = cumreturn / counter_rollouts
            print(f'avg return: {avgreturn}')
            counter_rollouts = 0
            cumreturn = 0.
            if avgreturn > 1.0 - EPS:
                break

    pass