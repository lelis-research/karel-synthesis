import logging
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from gym.spaces.box import Box
from dsl.production import Production
from embedding.autoencoder.program_vae import ProgramVAE
from config.config import Config
from embedding.models.SupervisedModel import SupervisedModel
from embedding.program_dataset import make_datasets
from karel.vec_env import VecEnv

DATA_DIR = 'data/program_dataset'

config = {
    'algorithm': 'supervised',                      # current training algorithm: 'supervised', 'RL', 'supervisedRL', 'debug', 'output_dataset_split'
    'net': {
        'saved_params_path': None,                  # path to load saved weights in a loaded network
        'saved_sup_params_path': None,              # path to load saved weights from supervised training
        'rnn_type': 'GRU',                          # recurrent unit type
        'decoder': {
            'use_teacher_enforcing': True,          # teacher enforcing while SL training
            'freeze_params': False                  # freeze decoder params if set True
        },
        'condition':{
            'freeze_params': False,
            'use_teacher_enforcing': True,
            'observations': 'environment',          # condition policy input from ['environment', 'dataset', 'initial_state']
        },
        'controller':{
            'add_noise': False,                     # add nosie to meta-controller weights like StyleGAN
            'input_coef': 0.01,                     # if using constant vector as input to controller, use this as multiplier
            'use_decoder_dist': True,               # if True, RL on decoder distribution, otherwise RL on meta-controller distribution
            'use_previous_programs': False,         # if True, use previous program as input to meta-controller
            'program_reduction': 'identity',        # 'identity': no-reduction, 'mean': mean of all previous program as input to meta-controller
        },
        'tanh_after_mu_sigma': False,               # apply tanh after distribution (mean and std of VAE) layers
        'tanh_after_sample': False,                 # apply tanh after sampling from VAE distribution
    },
    'rl':{
        'num_processes': 64,                        # how many training CPU processes to use (default: 32)
        'num_steps': 8,                             # 'number of forward steps (default: 32)'
        'num_env_steps': 10e6,                      # 'number of environment steps to train (default: 10e6)'
        'gamma': 0.99,                              # discount factor for rewards (default: 0.99)
        'use_gae': True,                            # 'use generalized advantage estimation'
        'gae_lambda': 0.95,                         # 'gae lambda parameter (default: 0.95)'
        'use_proper_time_limits': False,            # 'compute returns taking into account time limits'
        'use_all_programs': False,                  # 'False sets all mask value to 1 (ignores done_ variable value in trainer.py)'
        'future_rewards': False,                    # True: Maximizing expected future reward, False: Maximizing current reward
        'value_method': 'mean',                     # mean: mean of token values, program_embedding: value of eop token
        'envs': {
            'executable': {
                'name': 'karel',
                'task_definition': 'program',       # choices=['program', 'custom_reward']
                'task_file': 'tasks/test1.txt',  # choose from these tokens to write a space separated VALID program
                # for ground_truth task: ['DEF', 'run', 'm(', 'm)', 'move', 'turnRight',
                # 'turnLeft', 'pickMarker', 'putMarker', 'r(', 'r)', 'R=0', 'R=1', 'R=2',
                # 'R=3', 'R=4', 'R=5', 'R=6', 'R=7', 'R=8', 'R=9', 'R=10', 'R=11', 'R=12',
                # 'R=13', 'R=14', 'R=15', 'R=16', 'R=17', 'R=18', 'R=19', 'REPEAT', 'c(',
                # 'c)', 'i(', 'i)', 'e(', 'e)', 'IF', 'IFELSE', 'ELSE', 'frontIsClear',
                # 'leftIsClear', 'rightIsClear', 'markersPresent', 'noMarkersPresent',
                # 'not', 'w(', 'w)', 'WHILE']
                'max_demo_length': 100,             # maximum demonstration length
                'min_demo_length': 1,               # minimum demonstration length
                'num_demo_per_program': 10,         # 'number of seen demonstrations'
                'dense_execution_reward': False,    # encode reward along with state and action if task defined by custom reward
            },
            'program': {
                'mdp_type': 'ProgramEnv1',          # choices=['ProgramEnv1']
                'intrinsic_reward': False,          # NGU paper based intrinsic reward
                'intrinsic_beta': 0.0,              # reward = env_reward + intrinsic_beta * intrinsic_reward
            }
        },
        'policy':{
          'execution_guided': False,                # 'enable execution guided program synthesis'
          'two_head': False,                        # 'predict end-of-program token separate than program tokens'
          'recurrent_policy': True,                 # 'use a recurrent policy'
        },
        'algo':{
            'name': 'reinforce',
            'value_loss_coef':0.5,                  # 'value loss coefficient (default: 0.5)'
            'entropy_coef':0.1,                     # 'entropy term coefficient (default: 0.01)'
            'final_entropy_coef': 0.01,             # 'final entropy term coefficient (default: None)'
            'use_exp_ent_decay': False,             # 'use a exponential decay schedule on the entropy coef'
            'use_recurrent_generator': False,       # 'use episodic memory replay'
            'max_grad_norm': 0.5,                   # 'max norm of gradients (default: 0.5)'
            'lr': 5e-4,                             # 'learning rate (default: 5e-4)'
            'use_linear_lr_decay': True,            # 'use a linear schedule on the learning rate'
            'ppo':{
                'clip_param':0.1,                   # 'ppo clip parameter (default: 0.1)'
                'ppo_epoch':2,                      # 'number of ppo epochs (default: 4)'
                'num_mini_batch':2,                 # 'number of batches for ppo (default: 4)'
                'eps': 1e-5,                        # 'RMSprop optimizer epsilon (default: 1e-5)'
            },
            'a2c':{
                'eps': 1e-5,                        # 'RMSprop optimizer epsilon (default: 1e-5)'
                'alpha': 0.99,                      # 'RMSprop optimizer apha (default: 0.99)'
            },
            'acktr':{
            },
            'reinforce': {
                'clip_param': 0.1,                  # 'ppo clip parameter (default: 0.1)'
                'reinforce_epoch': 1,               # 'number of ppo epochs (default: 4)'
                'num_mini_batch': 2,                # 'number of batches for ppo (default: 4)'
                'eps': 1e-5,                        # 'RMSprop optimizer epsilon (default: 1e-5)'
            },
        },
        'loss':{
                'decoder_rl_loss_coef': 1.0,            # coefficient of policy loss during RL training
                'condition_rl_loss_coef': 0.0,          # coefficient of condition network loss during RL training
                'latent_rl_loss_coef': 0.0,             # coefficient of latent loss (beta) in VAE during RL training
                'use_mean_only_for_latent_loss': False, # applying latent loss only to mean while searching over latent space
            }
    },
    'train': {
        'data': {                                   # Dictionary to control dataset characteristics
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 256,
        'shuffle': True,
        'max_epoch': 150,
    },
    'valid': {
        'data': {                                   # Dictionary to control dataset characteristics
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 256,
        'shuffle': True,
        'debug_samples': [3, 37, 54],               # sample ids to generate plots for (None, int, list)
    },
    'loss': {
        'latent_loss_coef': 1.0,                    # coefficient of latent loss (beta) in VAE during SL training
        'condition_loss_coef': 1.0,                 # coefficient of condition policy loss during SL training
    },
    'dsl': {
        'max_program_len': 45,                      # maximum program length
    },
    'data_loader': {
        'num_workers': 0,                           # Number of parallel CPU workers
        'pin_memory': False,                        # Copy tensors into CUDA pinned memory before returning them
        'drop_last': True,
    },
    'optimizer': {
        'name': 'adam',
        'params': {
            'lr': 5e-4,
        },
        'scheduler': {
            'step_size': 10,                        # Period of learning rate decay
            'gamma': .95,                           # Multiplicative factor of learning rate decay
        }
    },
    'dsl_tokens': [
        'DEF', 'run', 'm(', 'm)', 'move', 'turnRight',
        'turnLeft', 'pickMarker', 'putMarker', 'r(', 'r)', 'R=0', 'R=1', 'R=2',
        'R=3', 'R=4', 'R=5', 'R=6', 'R=7', 'R=8', 'R=9', 'R=10', 'R=11', 'R=12',
        'R=13', 'R=14', 'R=15', 'R=16', 'R=17', 'R=18', 'R=19', 'REPEAT', 'c(',
        'c)', 'i(', 'i)', 'e(', 'e)', 'IF', 'IFELSE', 'ELSE', 'frontIsClear',
        'leftIsClear', 'rightIsClear', 'markersPresent', 'noMarkersPresent',
        'not', 'w(', 'w)', 'WHILE'
    ],
    'device': 'cpu',
    'height': 8,                                    # height of karel environment
    'width': 8,                                     # width of karel environment
    'num_lstm_cell_units': 64,                      # RNN latent space size
    'recurrent_policy': True,                       # If True, use RNN in policy network
    'two_head': False,                              # do we want two headed policy? Not for LEAPS
    'AE': False,                                    # using plain AutoEncoder instead of VAE
    'max_demo_length': 100,                         # maximum demonstration length (repeated)
    'grammar':'handwritten',                        # grammar type: [None, 'handwritten']
    'debug': True,
    'outdir': 'output/embeddings/',
    'record_file': 'records.pkl'
}


def main():

    device = torch.device('cpu')

    log_handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler('output/logs/stdout.txt', mode='w')]
    logging.basicConfig(handlers=log_handlers, format='%(asctime)s: %(message)s', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    writer = SummaryWriter(logdir='output/logs')

    dsl = Production.default_karel_production()

    np.random.seed(123)
    torch.manual_seed(123)
    random.seed(123)

    num_agent_actions = len(dsl.get_actions()) + 1
    config['dsl']['num_agent_actions'] = num_agent_actions

    global_logs = {'info': {}, 'result': {}}

    env = VecEnv(action_space=Box(low=0, high=51, shape=(45,), dtype=np.int16))

    model = ProgramVAE(dsl, Config(hidden_size=8))

    train_model = SupervisedModel(model, device, config, env, dsl, logger, writer, global_logs, True)

    p_train_dataset, p_val_dataset, p_test_dataset = make_datasets(
        DATA_DIR, config['dsl']['max_program_len'], config['max_demo_length'], 
        train_model.num_program_tokens, num_agent_actions, device, logger)

    p_train_dataloader = DataLoader(p_train_dataset, batch_size=16, shuffle=True, **config['data_loader'])
    p_val_dataloader = DataLoader(p_val_dataset, batch_size=16, shuffle=True, **config['data_loader'])
    p_test_dataloader = DataLoader(p_test_dataset, batch_size=16, shuffle=True, **config['data_loader'])

    r_train_dataloader = DataLoader(p_train_dataset, batch_size=config['rl']['num_steps'] * config['rl']['num_processes'],
                                    shuffle=True, **config['data_loader'])
    r_val_dataloader = DataLoader(p_val_dataset, batch_size=config['rl']['num_steps'] * config['rl']['num_processes'],
                                  shuffle=True, **config['data_loader'])

    train_model.train(p_train_dataloader, p_val_dataloader, r_train_dataloader, r_val_dataloader,
                max_epoch=config['train']['max_epoch'])

    train_model.evaluate(p_val_dataloader)

    train_model.evaluate(p_test_dataloader)

if __name__ == '__main__':

    main()
