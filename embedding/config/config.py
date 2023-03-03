from typing import NamedTuple


class Config(NamedTuple):
    batch_size: int = 256
    hidden_size: int = 64
    max_program_len: int = 45
    num_demo_per_program: int = 10
    env_height: int = 8
    env_width: int = 8
    max_demo_length: int = 100
    search_population_size: int = 256
    search_elitism_rate: float = 0.2