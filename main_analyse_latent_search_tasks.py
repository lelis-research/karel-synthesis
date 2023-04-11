import glob
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

models = [
    'leaps_vae_128',
    'sketch_vae_128_original_progs_n2',
    'sketch_vae_128_original_progs_n3',
    'sketch_vae_128_original_progs_n4'
]

models_labels = [
    'Leaps VAE (128)',
    'Sketch VAE (128), n=2',
    'Sketch VAE (128), n=3',
    'Sketch VAE (128), n=4'
]

tasks = [
    'StairClimberSparse',
    'MazeSparse',
    'FourCorners',
    'Harvester',
    'CleanHouse',
    'TopOff'
]

behaviour_types = [
    'SB', 'LB', 'CR'
]

behaviour_types_labels = [
    'Standard Environment',
    'Leaps Environment',
    'Crashable Environment'
]

output_folder = 'output_cedar'

def load_data(directory):
    
    seeds = glob.glob(f'{directory}/seed_*.csv')
    
    list_num_iterations = []
    list_best_rewards = []
    list_num_evaluations = []

    for seed in seeds:

        log = pd.read_csv(seed)
        
        num_iterations = log['iteration'].max()
        best_reward = log['best_reward'].max()
        total_num_evaluations = log['num_evaluations'].sum()
        
        list_num_iterations.append(num_iterations)
        list_best_rewards.append(best_reward)
        list_num_evaluations.append(total_num_evaluations)
    
    return list_num_iterations, list_best_rewards, list_num_evaluations

if __name__ == '__main__':

    os.makedirs(f'{output_folder}/plots', exist_ok=True)

    for behaviour_type, behaviour_type_label in zip(behaviour_types, behaviour_types_labels):
        
        for task in tasks:
        
            num_iterations = []
            best_rewards = []
            num_evaluations = []
            
            print()
            print(f'{task}, {behaviour_type_label}')
        
            for model in models:

                directory = f'{output_folder}/{model}_{task}_{behaviour_type}/latent_search'
                model_num_iterations, model_best_reward, model_num_evaluations = load_data(directory)
                
                num_iterations.append(model_num_iterations)
                best_rewards.append(model_best_reward)
                num_evaluations.append(model_num_evaluations)
                
                print('Model', model)
                print('Best reward', np.mean(model_best_reward))
                print('Evaluations', np.mean(model_num_evaluations))
                print()

            fig = plt.figure(figsize=(5,5))
            fig.suptitle(f'{task}, {behaviour_type_label}')
            
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.boxplot(best_rewards, labels=models_labels)
            ax1.set_ylim([-1.1, 1.1])
            ax1.set_ylabel('Best reward')
            plt.setp(ax1.get_xticklabels(), visible=False)
            # ax1.set_xlabel('Model')

            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
            ax2.boxplot(num_evaluations, labels=models_labels)
            ax2.set_ylabel('Number of evaluations')
            ax2.xaxis.set_tick_params(rotation=30)
            # ax2.set_xlabel('Model')
            
            fig.tight_layout()
            fig.savefig(f'{output_folder}/plots/{task}_{behaviour_type}.png')
            plt.close()
