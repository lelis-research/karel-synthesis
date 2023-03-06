import os
from matplotlib import pyplot as plt
import pandas as pd

model = 'leaps_vae_64'

os.makedirs(f'output/{model}/plots')

train_log = pd.read_csv(f'output/{model}/training_info.csv')
val_log = pd.read_csv(f'output/{model}/validation_info.csv')

logs = pd.merge(train_log, val_log, on='epoch', suffixes=['_train', '_val'])

logs[['mean_total_loss_train', 'mean_progs_loss_train', 'mean_a_h_loss_train', 'mean_latent_loss_train']].plot.line(grid=True)
plt.savefig(f'output/{model}/plots/mean_loss_train.png')
plt.close()

logs[['mean_total_loss_val', 'mean_progs_loss_val', 'mean_a_h_loss_val', 'mean_latent_loss_val']].plot.line(grid=True)
plt.savefig(f'output/{model}/plots/mean_loss_val.png')
plt.close()

logs[['mean_progs_t_accuracy_train', 'mean_progs_s_accuracy_train', 'mean_a_h_t_accuracy_train', 'mean_a_h_s_accuracy_train']].plot.line(grid=True)
plt.savefig(f'output/{model}/plots/mean_acc_train.png')
plt.close()

logs[['mean_progs_t_accuracy_val', 'mean_progs_s_accuracy_val', 'mean_a_h_t_accuracy_val', 'mean_a_h_s_accuracy_val']].plot.line(grid=True)
plt.savefig(f'output/{model}/plots/mean_acc_val.png')
plt.close()
