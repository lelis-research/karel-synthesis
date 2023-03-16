import torch
from config.config import Config
from dsl.production import Production
from vae.models.leaps_vae import LeapsVAE
from search.latent_search import LatentSearch
from tasks.stair_climber import StairClimber


if __name__ == '__main__':
    
    dsl = Production.default_karel_production()

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LeapsVAE(dsl, device)
    
    task = StairClimber()
    
    params = torch.load(f'params/leaps_vae_256.ptp', map_location=device)
    model.load_state_dict(params, strict=False)
    
    searcher = LatentSearch(model, task)
    
    searcher.search()
