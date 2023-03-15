import torch
from config.config import Config
from dsl.production import Production
from vae.models.leaps_vae import LeapsVAE
from vae.models.policy_vae import PolicyVAE
from search.latent_search import LatentSearch
from tasks.stair_climber import StairClimber


if __name__ == '__main__':
    
    dsl = Production.default_karel_production()

    device = torch.device('cpu')
    
    Config.model_hidden_size = 32
    # Config.search_reduce_to_mean = True
    Config.env_enable_leaps_behaviour = True
    Config.env_is_crashable = False
    
    model = LeapsVAE(dsl, device)
    
    task = StairClimber()
    
    params = torch.load(f'output/leaps_vae_debug/model/best_val.ptp', map_location=device)
    model.load_state_dict(params, strict=False)
    
    searcher = LatentSearch(model, task)
    
    searcher.search()
