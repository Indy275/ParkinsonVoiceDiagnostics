import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('settings.ini')
plot_results = config.getboolean('OUTPUT_SETTINGS', 'plot_results')

if os.getenv("COLAB_RELEASE_TAG"):  # colab
    experiment_folder = '/content/drive/My Drive/RAIVD_data/experiments/'
else:
    cwd = os.path.abspath(os.getcwd())
    experiment_folder = os.path.join(cwd,'experiments')

parent = os.path.dirname
path = parent(parent(__file__))

def make_plot(ax, metrics_df, title, color='blue', lab=None):
    lbl = lab if lab else 'Target set'
    ax.plot(metrics_df['Iteration'], metrics_df['ROC_AUC'], label=lbl, marker='.', alpha=0.8, color=color, linewidth=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)
    ax.set_ylim((0.4,1.0))
    return ax


def plot_TL_performance(base_dataset, target_dataset, ifm_clf, nifm_clf, ax=None, firstcol=True, lastrow=True, legend=True):
    nifm_model = 'hubert0'
    try:
        metrics_ifm = pd.read_csv(os.path.join(experiment_folder, f'{ifm_clf}_ifm_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
        base_ifm = pd.read_csv(os.path.join(experiment_folder, f'{ifm_clf}_ifm_metrics_{base_dataset}_{target_dataset}_base.csv'))
        print(os.path.join(experiment_folder, f'{ifm_clf}_ifm_metrics_{base_dataset}_{target_dataset}_base.csv'))
    except:
        print("IFM model not found, trying DNN")
        ifm_clf = 'DNN' if ifm_clf == 'SGD' else 'SGD'
        metrics_ifm = pd.read_csv(os.path.join(experiment_folder, f'{ifm_clf}_ifm_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
        base_ifm = pd.read_csv(os.path.join(experiment_folder, f'{ifm_clf}_ifm_metrics_{base_dataset}_{target_dataset}_base.csv'))
    try:
        metrics_nifm = pd.read_csv(os.path.join(experiment_folder, f'{nifm_clf}_{nifm_model}_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
        base_nifm = pd.read_csv(os.path.join(experiment_folder, f'{nifm_clf}_{nifm_model}_metrics_{base_dataset}_{target_dataset}_base.csv'))
        print(os.path.join(experiment_folder, f'{nifm_clf}_{nifm_model}_metrics_{base_dataset}_{target_dataset}_base.csv'))

    except:
        print("NIFM model not found, trying DNN")
        nifm_clf = 'DNN' if nifm_clf == 'SGD' else 'SGD'
        metrics_nifm = pd.read_csv(os.path.join(experiment_folder, f'{nifm_clf}_{nifm_model}_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
        base_nifm = pd.read_csv(os.path.join(experiment_folder, f'{nifm_clf}_{nifm_model}_metrics_{base_dataset}_{target_dataset}_base.csv'))

    mono = pd.read_csv(os.path.join(experiment_folder, f'monolingual_result.csv'))
    ds = target_dataset.split('_')
    tgt_mono_nifm = mono[(mono['dataset']==ds[0]) & (mono['task']==ds[1]) & (mono['ifm_nifm']==nifm_model)]
    tgt_mono_ifm = mono[(mono['dataset']==ds[0]) & (mono['task']==ds[1])  & (mono['ifm_nifm']=='ifm')]


    if ax is None:
        ax2 = plt.subplot(1,1,1)
    else:
        ax2 = ax

    colors = plt.cm.tab20b(np.linspace(0, 1, 20))
    c1 = colors[1]
    c2 = colors[13]

    if firstcol:
        ax2.set_ylabel('Performance (AUC)')
    if lastrow:
        ax2.set_xlabel('N-shots')
        
    title = f"{'+'.join(base_dataset.split('_')[:-1])} -> {target_dataset.replace('_', ' ')}" # {base_dataset.split('_')[-1]} 
    make_plot(ax2, metrics_nifm, title, lab='Target set NIFM', color=c1)
    make_plot(ax2, metrics_ifm, title, lab='Target set IFM', color=c2)

    ax2.axhline(y=tgt_mono_nifm['AUC'].iloc[-1], label='Target set NIFM baseline', linestyle='dotted', color=c1,linewidth=1)
    ax2.axhline(y=tgt_mono_ifm['AUC'].iloc[-1], label='Target set IFM baseline', linestyle='dotted', color=c2, linewidth=1)
    ax2.axhline(y=0.5, label='Random guessing', linestyle='dotted', color='k', linewidth=1)

   
    ax2.plot(base_nifm['Iteration'], base_nifm['ROC_AUC'], label='Base set NIFM', linestyle='dashed', color=c1 , marker='x', alpha=0.7, linewidth=1)
    ax2.plot(base_ifm['Iteration'], base_ifm['ROC_AUC'], label='Base set IFM', linestyle='dashed',color=c2, marker='x', alpha=0.7, linewidth=1)

    if legend:
        leg = ax2.legend(bbox_to_anchor=(0.8, 2.), bbox_transform=ax2.transAxes)
        leg = ax2.legend(bbox_to_anchor=(1, 0.), loc='lower right', bbox_transform=ax2.transAxes)

        return ax2, leg

    plt.tight_layout()
    if plot_results:
        plt.show()
    return ax2


if __name__ == "__main__":
    ifm_clf, nifm_clf = 'SGD', 'SGD'
    base = 'NeuroVoz_PCGITA_IPVS_tdu'
    tgt = 'MDVR_tdu'
    plot, legend = plot_TL_performance(base, tgt, ifm_clf, nifm_clf)
    plot.add_artist(legend)
    
    plt.savefig(os.path.join(experiment_folder, 'FSTL_example_MDVR_tdu.pdf'))