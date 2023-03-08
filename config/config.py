class Config:

    model_name: str = 'program_vae'
    model_hidden_size: int = 256
    
    data_batch_size: int = 256
    data_max_program_len: int = 45
    data_max_demo_length: int = 100
    data_num_demo_per_program: int = 10
    data_ratio_train: float = 0.7
    data_ratio_val: float = 0.15
    data_ratio_test: float = 0.15
    
    env_task: str = 'StairClimber'
    env_seed: int = 1
    env_height: int = 8
    env_width: int = 8
    env_leaps_behaviour: bool = False
    env_crashable: bool = True
    
    search_elitism_rate: float = 0.1
    search_population_size: int = 256
    search_sigma: float = 0.2
    search_number_executions: int = 100
    search_number_iterations: int = 1000
    
    trainer_num_epochs: int = 150
    trainer_prog_teacher_enforcing: bool = True
    trainer_a_h_teacher_enforcing: bool = True
    trainer_prog_loss_coef: float = 1.0
    trainer_a_h_loss_coef: float = 1.0
    trainer_latent_loss_coef: float = 0.1
    trainer_optim_lr: float = 5e-4
    
    @classmethod
    def parse_args(cls):
        """Generates an argparser for setting all Config attributes through command-line.
        """        
        from argparse import ArgumentParser
        
        parser = ArgumentParser()
        for param_name in cls.__annotations__:
            parser.add_argument(f'--{param_name}',
                                default=cls.__dict__[param_name],
                                type=cls.__annotations__[param_name])
        
        args_dict = vars(parser.parse_args())
        
        for param_name in cls.__annotations__:
            if cls.__dict__[param_name] != args_dict[param_name]:
                setattr(cls, param_name, args_dict[param_name])
        