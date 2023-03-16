import torch
from dsl.production import Production
from vae.models.leaps_vae import LeapsVAE
from search.latent_search import LatentSearch
from tasks.stair_climber import StairClimber


if __name__ == '__main__':
    
    dsl = Production.default_karel_production()

    device = torch.device('cuda')
    
    model = LeapsVAE(dsl, device)
    
    task = StairClimber()
    
    params = torch.load(f'params/leaps_vae_256.ptp', map_location=device)
    model.load_state_dict(params, strict=False)
    
    searcher = LatentSearch(model, task)
    
    searcher.search()
