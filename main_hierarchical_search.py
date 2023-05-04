import torch
from config import Config
from dsl import DSL
from logger.stdout_logger import StdoutLogger
from search.hierarchical_search import HierarchicalSearch
from search.hierarchical_search_mab import HierarchicalSearchMAB
from vae.models import load_model
from tasks import get_task_cls


if __name__ == '__main__':
    
    dsl = DSL.init_default_karel()

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sketch_model = load_model(Config.hierarchical_model_level_1_class, dsl, device, Config.hierarchical_model_level_1_hidden_size)
    sketch_params = torch.load(Config.hierarchical_model_level_1_params_path, map_location=device)
    sketch_model.load_state_dict(sketch_params, strict=False)

    holes_model = load_model(Config.hierarchical_model_level_2_class, dsl, device, Config.hierarchical_model_level_2_hidden_size)
    holes_params = torch.load(Config.hierarchical_model_level_2_params_path, map_location=device)
    holes_model.load_state_dict(holes_params, strict=False)
    
    models = [sketch_model, holes_model]
    
    task_cls = get_task_cls(Config.env_task)
    
    if Config.hierarchical_search_mode == 'DFS':
        searcher = HierarchicalSearch(models, task_cls, dsl)
    else:
        searcher = HierarchicalSearchMAB(models, task_cls, dsl)
    
    StdoutLogger.log('Main', f'Starting Hierarchical Search ({Config.hierarchical_search_mode}) with models:')
    StdoutLogger.log('Main', f'{Config.hierarchical_model_level_1_class} ({Config.hierarchical_model_level_1_hidden_size}) and')
    StdoutLogger.log('Main', f'{Config.hierarchical_model_level_2_class} ({Config.hierarchical_model_level_2_hidden_size})')
    StdoutLogger.log('Main', f'For task {Config.env_task}.')
    
    best_program, converged, num_evaluations = searcher.search()
    
    StdoutLogger.log('Main', f'Converged: {converged}')
    StdoutLogger.log('Main', f'Final program: {best_program}')
    StdoutLogger.log('Main', f'Number of evaluations: {num_evaluations}')
