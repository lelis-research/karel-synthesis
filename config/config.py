from typing import Annotated

class Config:
    """Class that handles the project global configuration.
    """

    disable_gpu: Annotated[bool, 'Disable GPU, even if available. Useful for debugging.'] = False

    dsl_include_hole: Annotated[bool, 'If set, DSL includes <HOLE> token.'] = False

    experiment_name: Annotated[str, 'Name of the model, used for saving output.'] = 'program_vae'

    multiprocessing_active: Annotated[bool, 'If set, search functions will use multiprocessing to evaluate programs.'] = False

    model_name: Annotated[str, 'Class name of the VAE model.'] = 'LeapsVAE'
    model_seed: Annotated[int, 'Seed for model initialization.'] = 1
    model_hidden_size: Annotated[int, 'Number of dimensions in VAE hidden unit.'] = 256
    model_params_path: Annotated[str, 'Path to model parameters.'] = 'params/leaps_vae_256.ptp'
    
    hierarchical_levels: Annotated[int, 'Number of levels in the hierarchical search.'] = 2
    hierarchical_search_mode: Annotated[str, 'Search mode for hierarchical search. Can be "MAB" or "BFS".'] = 'BFS'
    hierarchical_model_level_1_class: Annotated[str, 'Class name of the model for level 1.'] = 'SketchVAE'
    hierarchical_model_level_1_hidden_size: Annotated[int, 'Number of dimensions in level 1 model hidden unit.'] = 128
    hierarchical_model_level_1_params_path: Annotated[str, 'Path to model parameters for level 1.'] = 'params/sketch_vae_128.ptp'
    hierarchical_model_level_2_class: Annotated[str, 'Class name of the model for level 2.'] = 'LeapsVAE'
    hierarchical_model_level_2_hidden_size: Annotated[int, 'Number of dimensions in level 2 model hidden unit.'] = 32
    hierarchical_model_level_2_params_path: Annotated[str, 'Path to model parameters for level 2.'] = 'params/leaps_vae_32.ptp'

    hierarchical_level_1_pop_size: Annotated[int, 'Population size for level 1 search.'] = 8
    hierarchical_level_2_pop_size: Annotated[int, 'Population size for level 2 search.'] = 16
    hierarchical_level_1_iterations: Annotated[int, 'Number of iterations for level 1 search.'] = 3
    hierarchical_level_2_iterations: Annotated[int, 'Number of iterations for level 2 search.'] = 3
    
    hierarchical_mab_epsilon: Annotated[float, 'Epsilon for MAB action selection.'] = 0.1
    hierarchical_mab_batch_size: Annotated[int, 'Batch size for MAB sampling.'] = 64
    hierarchical_mab_search_iterations: Annotated[int, 'Number of iterations for MAB search.'] = 3
    hierarchical_mab_sample_iterations: Annotated[int, 'Number of iterations for MAB program sampling.'] = 3
    
    datagen_num_programs: Annotated[int, 'Number of programs in dataset, used for data generation and loading.'] = 50000
    datagen_sketch_iterations: Annotated[int, 'Number of needed Top-Down iterations to reconstruct a program from its sketch'] = 3
    datagen_generate_demos: Annotated[bool, 'If set, generates demonstrations for each program.'] = False
    datagen_generate_sketches: Annotated[bool, 'If set, generates sketches for each program.'] = False
    
    data_class_name: Annotated[str, 'Name of program dataset class.'] = 'ProgramDataset'
    data_program_dataset_path: Annotated[str, 'Path to program dataset.'] = 'data/programs.pkl'
    data_reduce_dataset: Annotated[bool, 'Reduce dataset to 1000 samples for debugging'] = False
    data_batch_size: Annotated[int, 'Batch size used in VAE training.'] = 256
    data_max_program_length: Annotated[int, 'Maximum program length in number of tokens.'] = 45
    data_max_program_size: Annotated[int, 'Maximum program size in number of nodes.'] = 20
    data_max_program_depth: Annotated[int, 'Max allowed depth during program generation.'] = 4
    data_max_program_sequence: Annotated[int, 'Max allowed number of sequential nodes aggregated by Conjunction.'] = 6
    data_max_demo_length: Annotated[int, 'Maximum action history length in number of actions.'] = 100
    data_num_demo_per_program: Annotated[int, 'Number of demonstrations per program in dataset.'] = 10
    data_ratio_train: Annotated[float, 'Ratio of training data.'] = 0.7
    data_ratio_val: Annotated[float, 'Ratio of validation data.'] = 0.15
    data_ratio_test: Annotated[float, 'Ratio of test data.'] = 0.15
    
    env_task: Annotated[str, 'Name of Karel task to solve.'] = 'StairClimber'
    env_seed: Annotated[int, 'Seed for random environment generation.'] = 1
    env_height: Annotated[int, 'Height of Karel environment.'] = 8
    env_width: Annotated[int, 'Width of Karel environment.'] = 8
    env_enable_leaps_behaviour: Annotated[bool, 'If set, uses LEAPS version of Karel rules.'] = False
    env_is_crashable: Annotated[bool, 'If set, program stops when Karel crashes.'] = False
    
    search_topdown_iterations: Annotated[int, 'Maximum iterations for Top-Down Search.'] = 3
    search_elitism_rate: Annotated[float, 'Elitism rate for selection phase of Latent Search.'] = 0.125
    search_population_size: Annotated[int, 'Population size for growth phase of Latent Search.'] = 256
    search_reduce_to_mean: Annotated[bool, 'If set, elite population is reduced to mean in each iteration'] = False
    search_sigma: Annotated[float, 'Size of noise in growth phase of Latent Search.'] = 0.2
    search_number_executions: Annotated[int, 'Number of environment executions for mean reward calculation.'] = 16
    search_number_iterations: Annotated[int, 'Maximum number of iterations of Latent Search.'] = 1000
    search_restart_timeout: Annotated[int, 'Maximum number of iterations without improvement before restart.'] = 5
    
    trainer_num_epochs: Annotated[int, 'Number of training epochs.'] = 150
    trainer_disable_prog_teacher_enforcing: Annotated[bool, 'If set, program sequence classification will not use teacher enforcing.'] = False
    trainer_disable_a_h_teacher_enforcing: Annotated[bool, 'If set, actions sequence classification will not use teacher enforcing.'] = False
    trainer_prog_loss_coef: Annotated[float, 'Weight of program classification loss.'] = 1.0
    trainer_a_h_loss_coef: Annotated[float, 'Weight of actions classification loss.'] = 1.0
    trainer_latent_loss_coef: Annotated[float, 'Weight of VAE KL Divergence Loss.'] = 0.1
    trainer_optim_lr: Annotated[float, 'Adam optimizer learning rate.'] = 5e-4
    trainer_save_params_each_epoch: Annotated[bool, 'If set, trainer saves model params after each epoch.'] = False
