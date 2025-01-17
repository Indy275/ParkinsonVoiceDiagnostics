import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os
import configparser

config = configparser.ConfigParser()
config.read('settings.ini')
plot_results = config.getboolean('OUTPUT_SETTINGS', 'plot_results')

parent = os.path.dirname
path = parent(parent(__file__))

def make_plot(ax, metrics_df, title, color='purple', lab=None, legend=True):
    lbl = lab if lab else 'Target set'
    ax.plot(metrics_df['Iteration'], metrics_df['ROC_AUC'], label=lbl, marker='o', alpha=0.8, color=color, linewidth=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title)
    ax.title.set_fontsize(10)
    if legend:
        ax.legend()
        ax.lines.pop(0)
    ax.grid(True)
    ax.set_ylim((0.4,1.0))

    return ax


def plot_TL_performance(base_dataset, target_dataset, clf_name, ax=None, firstcol=True, lastrow=True, legend=True):
    # Load dataframes for file-level, speaker-level, and base-level performance
    # metrics_df = pd.read_csv(os.path.join(path, 'experiments', f'{clf_name}_{ifm_or_nifm}_metrics_{base_dataset}_{target_dataset}.csv'))
    metrics_nifm = pd.read_csv(os.path.join(path, 'experiments', f'{clf_name}_vgg_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
    metrics_ifm = pd.read_csv(os.path.join(path, 'experiments', f'{clf_name}_ifm_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
    base_nifm = pd.read_csv(os.path.join(path, 'experiments', f'{clf_name}_vgg_metrics_{base_dataset}_{target_dataset}_base.csv'))
    base_ifm = pd.read_csv(os.path.join(path, 'experiments', f'{clf_name}_ifm_metrics_{base_dataset}_{target_dataset}_base.csv'))
    
    mono = pd.read_csv(os.path.join(path, 'experiments', f'monolingual_result.csv'))
    tgt_mono_ifm = mono[(mono['dataset']==target_dataset) & (mono['model']==clf_name) & (mono['ifm_nifm']=='ifm')]
    tgt_mono_nifm = mono[(mono['dataset']==target_dataset) & (mono['model']==clf_name) & (mono['ifm_nifm']=='vgg')]

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


    ax2.plot(base_nifm['Iteration'], base_nifm['ROC_AUC'], label='Base NIFM', linestyle='dashed', color='blue' , marker='.', alpha=0.8, linewidth=1)
    ax2.plot(base_ifm['Iteration'], base_ifm['ROC_AUC'], label='Base IFM', linestyle='dashed',color='red', marker='.', alpha=0.8, linewidth=1)

    if len(tgt_mono_ifm)>0 and len(tgt_mono_nifm)>0: # row exists
            monoscore = tgt_mono_ifm['sMAUC'].iloc[-1]
            ax2.axhline(y=monoscore, label='Target set monolingual IFM', linestyle='dotted', color='red',linewidth=2)
            monoscore = tgt_mono_nifm['sMAUC'].iloc[-1]
            ax2.axhline(y=monoscore, label='Target set monolingual NIFM', linestyle='dotted', color='blue',linewidth=2)
    else:
        print("Not in monolingual set:", target_dataset, clf_name, "IFM:",len(tgt_mono_ifm), "vgg:",len(tgt_mono_nifm))
    
    title = f"{'+'.join(base_dataset.split('_')[:-1])} \n -> {target_dataset.replace('_', ' ')}" # {base_dataset.split('_')[-1]} 
    make_plot(ax2, metrics_ifm, title, lab='Target set IFM', color='red', legend=legend)
    make_plot(ax2, metrics_nifm, title, lab='Target set NIFM', color='blue', legend=legend)

    plt.tight_layout()
    plt.savefig(os.path.join('experiments', f'{clf_name}_fstlcomp_{base_dataset}_{target_dataset}'), dpi=300)
    if plot_results:
        plt.show()
    return ax2


if __name__ == "__main__":
    clf_name = 'SGD'
    base = 'NeuroVozPCGITAIPVS_tdu'
    tgt = 'MDVR_tdu'
    plot_TL_performance(base, tgt, clf_name)
