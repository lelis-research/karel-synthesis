import torch
from config.config import Config
from dsl.production import Production
from embedding.autoencoder.leaps_vae import LeapsVAE
from embedding.autoencoder.policy_vae import PolicyVAE
from search.latent_search import LatentSearch
from tasks.stair_climber import StairClimber


if __name__ == '__main__':
    
    dsl = Production.default_karel_production()

    device = torch.device('cpu')
    
    Config.env_seed = 1
    
    model = PolicyVAE(dsl, device)
    
    task = StairClimber()
    
    params = torch.load(f'output/policy_vae_{Config.model_hidden_size}/model/best_val.ptp', map_location=device)
    model.load_state_dict(params, strict=False)
    
    searcher = LatentSearch(model, task)
    
    searcher.search()
