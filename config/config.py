from typing import Annotated, get_type_hints

class Config:
    """Class that handles the project global configuration.
    """

    model_name: Annotated[str, 'Name of the model, used for saving output.'] = 'program_vae'
    model_hidden_size: Annotated[int, 'Number of dimensions in hidden unit.'] = 256
    
    data_batch_size: Annotated[int, ''] = 256
    data_max_program_len: Annotated[int, ''] = 45
    data_max_demo_length: Annotated[int, ''] = 100
    data_num_demo_per_program: Annotated[int, ''] = 10
    data_ratio_train: Annotated[float, ''] = 0.7
    data_ratio_val: Annotated[float, ''] = 0.15
    data_ratio_test: Annotated[float, ''] = 0.15
    
    env_task: Annotated[str, ''] = 'StairClimber'
    env_seed: Annotated[int, ''] = 1
    env_height: Annotated[int, ''] = 8
    env_width: Annotated[int, ''] = 8
    env_leaps_behaviour: Annotated[bool, ''] = False
    env_crashable: Annotated[bool, ''] = True
    
    search_elitism_rate: Annotated[float, ''] = 0.1
    search_population_size: Annotated[int, ''] = 256
    search_sigma: Annotated[float, ''] = 0.2
    search_number_executions: Annotated[int, ''] = 100
    search_number_iterations: Annotated[int, ''] = 1000
    
    trainer_num_epochs: Annotated[int, ''] = 150
    trainer_prog_teacher_enforcing: Annotated[bool, ''] = True
    trainer_a_h_teacher_enforcing: Annotated[bool, ''] = True
    trainer_prog_loss_coef: Annotated[float, ''] = 1.0
    trainer_a_h_loss_coef: Annotated[float, ''] = 1.0
    trainer_latent_loss_coef: Annotated[float, ''] = 0.1
    trainer_optim_lr: Annotated[float, ''] = 5e-4
    
    @classmethod
    def parse_args(cls):
        """Generates an argparser for setting all Config attributes through command-line.
        """        
        from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
        
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        hints = get_type_hints(cls, include_extras=True)
        for param_name in cls.__annotations__:
            param_hints = hints.get(param_name).__dict__.get('__metadata__')
            joined_hints = ', '.join(param_hints) if param_hints else ''
            parser.add_argument(f'--{param_name}',
                                default=cls.__dict__[param_name],
                                type=cls.__annotations__[param_name],
                                help=joined_hints)
        
        args_dict = vars(parser.parse_args())
        
        for param_name in cls.__annotations__:
            if cls.__dict__[param_name] != args_dict[param_name]:
                setattr(cls, param_name, args_dict[param_name])
        