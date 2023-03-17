import torch
from config.config import Config
from dsl.production import Production
from logger.stdout_logger import StdoutLogger
from vae.models import load_model
from search.latent_search import LatentSearch
from tasks import get_task_cls


if __name__ == '__main__':
    
    dsl = Production.default_karel_production()

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(Config.model_name, dsl, device)
    
    task_cls = get_task_cls(Config.env_task)
    
    params = torch.load(f'params/leaps_vae_256.ptp', map_location=device)
    model.load_state_dict(params, strict=False)
    
    searcher = LatentSearch(model, task_cls)
    
    StdoutLogger.log('Main', f'Starting Latent Search with model {Config.model_name} for task {Config.env_task}')
    
    best_program, converged = searcher.search()
    
    StdoutLogger.log('Main', f'Converged: {converged}')
    StdoutLogger.log('Main', f'Final program: {best_program}')    
