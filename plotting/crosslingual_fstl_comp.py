import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os
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

def make_plot(ax, metrics_df, title, color='purple', lab=None):
    lbl = lab if lab else 'Target set'
    ax.plot(metrics_df['Iteration'], metrics_df['ROC_AUC'], label=lbl, marker='.', alpha=0.8, color=color, linewidth=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title)
    ax.title.set_fontsize(10)
    ax.set_ylim((0.4,1.0))
    return ax


def plot_TL_performance(base_dataset, target_dataset, ifm_clf, nifm_clf, ax=None, firstcol=True, lastrow=True, legend=True):
    nifm_model = 'hubert0'
    try:
        metrics_ifm = pd.read_csv(os.path.join(experiment_folder, f'{ifm_clf}_ifm_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
        base_ifm = pd.read_csv(os.path.join(experiment_folder, f'{ifm_clf}_ifm_metrics_{base_dataset}_{target_dataset}_base.csv'))
    except:
        ifm_clf = 'DNN' if ifm_clf == 'SGD' else 'SGD'
        metrics_ifm = pd.read_csv(os.path.join(experiment_folder, f'{ifm_clf}_ifm_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
        base_ifm = pd.read_csv(os.path.join(experiment_folder, f'{ifm_clf}_ifm_metrics_{base_dataset}_{target_dataset}_base.csv'))
    try:
        metrics_nifm = pd.read_csv(os.path.join(experiment_folder, f'{nifm_clf}_{nifm_model}_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
        base_nifm = pd.read_csv(os.path.join(experiment_folder, f'{nifm_clf}_{nifm_model}_metrics_{base_dataset}_{target_dataset}_base.csv'))
    except:
        nifm_clf = 'DNN' if nifm_clf == 'SGD' else 'SGD'
        metrics_nifm = pd.read_csv(os.path.join(experiment_folder, f'{nifm_clf}_{nifm_model}_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
        base_nifm = pd.read_csv(os.path.join(experiment_folder, f'{nifm_clf}_{nifm_model}_metrics_{base_dataset}_{target_dataset}_base.csv'))

    mono = pd.read_csv(os.path.join(experiment_folder, f'monolingual_result.csv'))
    tgt_mono_ifm = mono[(mono['dataset']==target_dataset) & (mono['model']==ifm_clf) & (mono['ifm_nifm']=='ifm')]
    tgt_mono_nifm = mono[(mono['dataset']==target_dataset) & (mono['model']==nifm_clf) & (mono['ifm_nifm']==nifm_model)]

    if ax is None:
        ax2 = plt.subplot(1,1,1)
    else:
        ax2 = ax

    if not firstcol:
        ax2.set_yticklabels([])
    else:
        ax.set_ylabel('Performance (AUC)')

    if not lastrow:
        ax2.set_xticklabels([])
    else:
        ax.set_xlabel('N-shots')

    # ax2.plot(base_nifm['Iteration'], base_nifm['ROC_AUC'], label='Base NIFM', linestyle='dashed', color='blue' , marker='x', alpha=0.8, linewidth=1)
    # ax2.plot(base_ifm['Iteration'], base_ifm['ROC_AUC'], label='Base IFM', linestyle='dashed',color='red', marker='x', alpha=0.8, linewidth=1)

    if len(tgt_mono_ifm)>0 and len(tgt_mono_nifm)>0: # row exists
            monoscore = tgt_mono_ifm['sMAUC'].iloc[-1]
            ax2.axhline(y=monoscore, label='Target set monolingual IFM', linestyle='dotted', color='red',linewidth=1)
            monoscore = tgt_mono_nifm['sMAUC'].iloc[-1]
            ax2.axhline(y=monoscore, label='Target set monolingual NIFM', linestyle='dotted', color='blue',linewidth=1)
    else:
        print("Not in monolingual set:", target_dataset,   "IFM:",len(tgt_mono_ifm), ifm_clf ,f"{nifm_model}:",len(tgt_mono_nifm), nifm_clf)
    
    title = f"{'+'.join(base_dataset.split('_')[:-1])} \n -> {target_dataset.replace('_', ' ')}" # {base_dataset.split('_')[-1]} 
    make_plot(ax2, metrics_ifm, title, lab='Target set IFM', color='red')
    make_plot(ax2, metrics_nifm, title, lab='Target set NIFM', color='blue')

    if legend:
        leg = ax2.legend(bbox_to_anchor=(1, 2.2), bbox_transform=ax2.transAxes)
        return ax2, leg

    plt.tight_layout()
    if plot_results:
        plt.show()
    return ax2


if __name__ == "__main__":
    clf_name = 'SGD'
    base = 'NeuroVozPCGITAIPVS_tdu'
    tgt = 'MDVR_tdu'
    plot_TL_performance(base, tgt, clf_name)
