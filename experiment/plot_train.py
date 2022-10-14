from matplotlib import pyplot as plt
import pandas as pd

train_log = pd.read_csv('output/experiment_training_data.csv')

train_log[['train_loss', 'val_loss']].plot.line(grid=True)
plt.savefig('output/experiment_plot_loss.png')
plt.close()

train_log[['train_acc', 'val_acc']].plot.line(grid=True)
plt.savefig('output/experiment_plot_acc.png')
plt.close()
