import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os
import configparser

config = configparser.ConfigParser()
config.read('settings.ini')

parent = os.path.dirname
path = parent(parent(__file__))


def make_plot(ax, metrics_df, title, color='red', lab=None):
    lbl = lab if lab else 'Target set'
    # ax.plot(metrics_df['Iteration'], metrics_df['Accuracy'], label='Target Accuracy', marker='o', alpha=0.8)
    ax.plot(metrics_df['Iteration'], metrics_df['ROC_AUC'], label=lbl, marker='o', alpha=0.8, color=color)
    # ax.plot(metrics_df['Iteration'], metrics_df['Sensitivity'], label='Sensitivity', marker='o')
    # ax.plot(metrics_df['Iteration'], metrics_df['Specificity'], label='Specificity', marker='o')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('N-shots')
    ax.set_ylabel('Performance (AUC)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    return ax


def plot_TL_performance(base_dataset, target_dataset, clf_name):
    # Load dataframes for file-level, speaker-level, and base-level performance
    # metrics_df = pd.read_csv(os.path.join(path, 'experiments', f'{clf_name}_{ifm_or_nifm}_metrics_{base_dataset}_{target_dataset}.csv'))
    metrics_nifm = pd.read_csv(os.path.join(path, 'experiments', f'{clf_name}_vgg_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
    metrics_ifm = pd.read_csv(os.path.join(path, 'experiments', f'{clf_name}_ifm_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
    # base_metrics = pd.read_csv(os.path.join(path, 'experiments', f'{clf_name}_{ifm_or_nifm}_metrics_{base_dataset}_{target_dataset}_base.csv'))
    
    mono = pd.read_csv(os.path.join(path, 'experiments', f'monolingual_result.csv'))
    if clf_name.endswith('FSTL'):
         clf_name = clf_name[:-4]
    tgt_mono_ifm = mono[(mono['dataset']==target_dataset) & (mono['model']==clf_name) & (mono['ifm_nifm']=='ifm')]
    tgt_mono_nifm = mono[(mono['dataset']==target_dataset) & (mono['model']==clf_name) & (mono['ifm_nifm']=='vgg')]

    ax2 = plt.subplot(1,1,1)

    # Plot Grouped (Speaker-Level) performance
    title = "Speaker-level classification performance against number of shots \n " \
            "Base: {} , Target: {}".format(base_dataset, target_dataset)
    # ax2.plot(base_metrics['Iteration'], base_metrics['ROC_AUC'], label='Base set', linestyle='dashed', marker='.')
    
    if len(tgt_mono_ifm)>0 and len(tgt_mono_nifm)>0: # row exists
            monoscore = tgt_mono_ifm['sMAUC'].iloc[-1]
            ax2.axhline(y=monoscore, label='Target set monolingual IFM', linestyle='dashed', color='orange')
            monoscore = tgt_mono_nifm['sMAUC'].iloc[-1]
            ax2.axhline(y=monoscore, label='Target set monolingual NIFM', linestyle='dashed', color='blue')
    else:
        print("Not in monolingual set:", target_dataset, clf_name, "IFM:",len(tgt_mono_ifm), "vgg:",len(tgt_mono_nifm))
    make_plot(ax2, metrics_ifm, title, lab='Target set IFM', color='orange')
    make_plot(ax2, metrics_nifm, title, lab='Target set NIFM', color='blue')

    plt.tight_layout()
    plt.savefig(os.path.join('experiments', f'{clf_name}_fstlcomp_{base_dataset}_{target_dataset}'), dpi=300)
    plt.show()


clf_name = 'SVMFSTL'
base = 'NeuroVozPCGITAItalianPD_tdu'
tgt = 'MDVR_tdu'
plot_TL_performance(base, tgt, clf_name)