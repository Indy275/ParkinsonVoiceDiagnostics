import matplotlib.pyplot as plt
import pandas as pd
import os
import configparser

config = configparser.ConfigParser()
config.read('settings.ini')
ifm_or_nifm = config['EXPERIMENT_SETTINGS']['ifm_or_nifm']
file_or_window = config['DATA_SETTINGS']['file_or_window']
if ifm_or_nifm == 'ifm':
    ifm_or_nifm += '_{}'.format(file_or_window)
clf_name = config['MODEL_SETTINGS']['clf']

parent = os.path.dirname
path = parent(parent(__file__))


def make_plot(ax, metrics_df, title):
    ax.plot(metrics_df['Iteration'], metrics_df['Accuracy'], label='Accuracy', marker='o')
    ax.plot(metrics_df['Iteration'], metrics_df['ROC_AUC'], label='ROC AUC', marker='o')
    ax.plot(metrics_df['Iteration'], metrics_df['Sensitivity'], label='Sensitivity', marker='o')
    ax.plot(metrics_df['Iteration'], metrics_df['Specificity'], label='Specificity', marker='o')

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Performance')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    return ax


def plot_TL_performance(base_dataset, target_dataset):
    fig = None
    if ifm_or_nifm[-4] == 'n':  # wiNdow or Nifm; majority voting only sensible for window-level predictions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        metrics_df = pd.read_csv(
            os.path.join(path, 'experiments\\{}metrics_{}_{}_grouped.csv'.format(clf_name,base_dataset, target_dataset)))
        title = "Sample-level Performance over iterations"
        make_plot(ax1, metrics_df, title)
    if not fig:
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    metrics_df = pd.read_csv(os.path.join(path, 'experiments\\{}metrics_{}_{}.csv'.format(clf_name,base_dataset, target_dataset)))
    title = 'Window-level Performance over iterations'
    make_plot(ax2, metrics_df, title)
    plt.tight_layout()
    plt.savefig('experiments\\metrics_{}_{}.png'.format(base_dataset, target_dataset), dpi=300)
    plt.show()
