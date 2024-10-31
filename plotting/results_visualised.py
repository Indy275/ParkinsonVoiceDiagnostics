import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os
import configparser

config = configparser.ConfigParser()
config.read('settings.ini')
ifm_or_nifm = config['EXPERIMENT_SETTINGS']['ifm_or_nifm']
clf_name = config['MODEL_SETTINGS']['clf']

parent = os.path.dirname
path = parent(parent(__file__))


def make_plot(ax, metrics_df, title):
    ax.plot(metrics_df['Iteration'], metrics_df['Accuracy'], label='Target Accuracy', marker='o', alpha=0.8)
    ax.plot(metrics_df['Iteration'], metrics_df['ROC_AUC'], label='Target ROC AUC', marker='o', alpha=0.8)
    # ax.plot(metrics_df['Iteration'], metrics_df['Sensitivity'], label='Sensitivity', marker='o')
    # ax.plot(metrics_df['Iteration'], metrics_df['Specificity'], label='Specificity', marker='o')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('N-shots')
    ax.set_ylabel('Performance')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    return ax


def plot_TL_performance(base_dataset, target_dataset):
    # Load dataframes for file-level, speaker-level, and base-level performance
    metrics_df = pd.read_csv(os.path.join(path, 'experiments', f'{clf_name}_{ifm_or_nifm}_metrics_{base_dataset}_{target_dataset}.csv'))
    metrics_grouped = pd.read_csv(os.path.join(path, 'experiments', f'{clf_name}_{ifm_or_nifm}_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
    base_metrics = pd.read_csv(os.path.join(path, 'experiments', f'{clf_name}_{ifm_or_nifm}_metrics_{base_dataset}_{target_dataset}_base.csv'))

    if base_dataset[-3:] != 'tdu':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot File/Window-Level performance
        title = f"File-level classification performance against number of shots \n  Base: {base_dataset} , Target: {target_dataset}"
        ax1.plot(base_metrics['Iteration'], base_metrics['ROC_AUC'], label='Base ROC AUC', linestyle='dashed', marker='.')
        make_plot(ax1, metrics_df, title)
    else:
        ax2 = plt.subplot(1,1,1)

    # Plot Grouped (Speaker-Level) performance
    title = "Speaker-level classification performance against number of shots \n " \
            "Base: {} , Target: {}".format(base_dataset, target_dataset)
    ax2.plot(base_metrics['Iteration'], base_metrics['ROC_AUC'], label='Base ROC AUC', linestyle='dashed', marker='.')

    make_plot(ax2, metrics_grouped, title)

    plt.tight_layout()
    plt.savefig(os.path.join('experiments', f'{clf_name}_{ifm_or_nifm}_metrics_{base_dataset}_{target_dataset}'), dpi=300)
    plt.show()
