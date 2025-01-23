import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os
import configparser

config = configparser.ConfigParser()
config.read('settings.ini')
ifm_or_nifm = config['EXPERIMENT_SETTINGS']['ifm_or_nifm']
clf_name = config['MODEL_SETTINGS']['clf']

if os.getenv("COLAB_RELEASE_TAG"):  # colab
    experiment_folder = '/content/drive/My Drive/RAIVD_data/experiments/'
else:
    cwd = os.path.abspath(os.getcwd())
    experiment_folder = os.path.join(cwd,'experiments')

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
    metrics_df = pd.read_csv(os.path.join(experiment_folder, f'{clf_name}_{ifm_or_nifm}_metrics_{base_dataset}_{target_dataset}.csv'))
    metrics_grouped = pd.read_csv(os.path.join(experiment_folder, f'{clf_name}_{ifm_or_nifm}_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
    base_metrics = pd.read_csv(os.path.join(experiment_folder, f'{clf_name}_{ifm_or_nifm}_metrics_{base_dataset}_{target_dataset}_base.csv'))
    mono = pd.read_csv(os.path.join(experiment_folder, f'monolingual_result.csv'))
    tgt_mono = mono[(mono['dataset']==target_dataset) & (mono['model']==clf_name[:3]) & (mono['ifm_nifm']==ifm_or_nifm)]

    if base_dataset[-3:] != 'tdu' and base_dataset[-3:] != 'ddk':  # file level preds can differ from speaker-level preds!
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot File/Window-Level performance
        title = f"File-level classification performance against number of shots \n  Base: {base_dataset} , Target: {target_dataset}"
        ax1.plot(base_metrics['Iteration'], base_metrics['ROC_AUC'], label='Base ROC AUC', linestyle='dashed', marker='.')
        if len(tgt_mono)>0: # row exists
            monoscore = tgt_mono['sMAUC'].iloc[-1]
            ax2.axhline(y=monoscore, label='Target AUC monolingual', linestyle='dashed', color='red')
        make_plot(ax1, metrics_df, title)
    else:
        ax2 = plt.subplot(1,1,1)

    # Plot Grouped (Speaker-Level) performance
    title = "Speaker-level classification performance against number of shots \n " \
            "Base: {} , Target: {}".format(base_dataset, target_dataset)
    ax2.plot(base_metrics['Iteration'], base_metrics['ROC_AUC'], label='Base ROC AUC', linestyle='dashed', marker='.')
    
    if len(tgt_mono)>0: # row exists
        monoscore = tgt_mono['sMAUC'].iloc[-1]
        ax2.axhline(y=monoscore, label='Target AUC monolingual', linestyle='dashed', color='red')
    else:
        print("Not in monolingual set:", target_dataset, clf_name, ifm_or_nifm)
    make_plot(ax2, metrics_grouped, title)

    plt.tight_layout()
    plt.savefig(os.path.join(experiment_folder, f'{clf_name}_{ifm_or_nifm}_metrics_{base_dataset}_{target_dataset}'), dpi=300)
    plt.show()


def fimp_plot(fimp, df):
    acoustic_feats = ['F0_mean', 'F0_std','F0_min', 'F0_max', 'dF0_mean', 'ddF0_mean', '%Jitter', 'absJitter', 'RAP', 'PPQ5', 'DDP', '%Shimmer', 'dbShimmer', 'APQ3', 'APQ5', 'APQ11', 'DDA','F1_mean','F2_mean','F3_mean', 'F1_bw_mean', 'F2_bw_mean', 'F3_bw_mean']
    mfcc_feats = ['mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean', 'mfcc5_mean', 'mfcc6_mean', 'mfcc7_mean', 'mfcc8_mean', 'mfcc9_mean', 'mfcc10_mean', 'mfcc11_mean', 'mfcc12_mean', 'mfcc13_mean', 'dmfcc1_mean', 'dmfcc2_mean', 'dmfcc3_mean', 'dmfcc4_mean', 'dmfcc5_mean', 'dmfcc6_mean', 'dmfcc7_mean', 'dmfcc8_mean', 'dmfcc9_mean', 'dmfcc10_mean', 'dmfcc11_mean', 'dmfcc12_mean', 'dmfcc13_mean', 'ddmfcc1_mean', 'ddmfcc2_mean', 'ddmfcc3_mean', 'ddmfcc4_mean', 'ddmfcc5_mean', 'ddmfcc6_mean', 'ddmfcc7_mean', 'ddmfcc8_mean', 'ddmfcc9_mean', 'ddmfcc10_mean', 'ddmfcc11_mean', 'ddmfcc12_mean', 'ddmfcc13_mean', 'mfcc1_std', 'mfcc2_std', 'mfcc3_std', 'mfcc4_std', 'mfcc5_std', 'mfcc6_std', 'mfcc7_std', 'mfcc8_std', 'mfcc9_std', 'mfcc10_std', 'mfcc11_std', 'mfcc12_std', 'mfcc13_std', 'dmfcc1_std', 'dmfcc2_std', 'dmfcc3_std', 'dmfcc4_std', 'dmfcc5_std', 'dmfcc6_std', 'dmfcc7_std', 'dmfcc8_std', 'dmfcc9_std', 'dmfcc10_std', 'dmfcc11_std', 'dmfcc12_std', 'dmfcc13_std', 'ddmfcc1_std', 'ddmfcc2_std', 'ddmfcc3_std', 'ddmfcc4_std', 'ddmfcc5_std', 'ddmfcc6_std', 'ddmfcc7_std', 'ddmfcc8_std', 'ddmfcc9_std', 'ddmfcc10_std', 'ddmfcc11_std', 'ddmfcc12_std', 'ddmfcc13_std', 'mfcc1_skew', 'mfcc2_skew', 'mfcc3_skew', 'mfcc4_skew', 'mfcc5_skew', 'mfcc6_skew', 'mfcc7_skew', 'mfcc8_skew', 'mfcc9_skew', 'mfcc10_skew', 'mfcc11_skew', 'mfcc12_skew', 'mfcc13_skew', 'dmfcc1_skew', 'dmfcc2_skew', 'dmfcc3_skew', 'dmfcc4_skew', 'dmfcc5_skew', 'dmfcc6_skew', 'dmfcc7_skew', 'dmfcc8_skew', 'dmfcc9_skew', 'dmfcc10_skew', 'dmfcc11_skew', 'dmfcc12_skew', 'dmfcc13_skew', 'ddmfcc1_skew', 'ddmfcc2_skew', 'ddmfcc3_skew', 'ddmfcc4_skew', 'ddmfcc5_skew', 'ddmfcc6_skew', 'ddmfcc7_skew', 'ddmfcc8_skew', 'ddmfcc9_skew', 'ddmfcc10_skew', 'ddmfcc11_skew', 'ddmfcc12_skew', 'ddmfcc13_skew', 'mfcc1_kurt', 'mfcc2_kurt', 'mfcc3_kurt', 'mfcc4_kurt', 'mfcc5_kurt', 'mfcc6_kurt', 'mfcc7_kurt', 'mfcc8_kurt', 'mfcc9_kurt', 'mfcc10_kurt', 'mfcc11_kurt', 'mfcc12_kurt', 'mfcc13_kurt', 'dmfcc1_kurt', 'dmfcc2_kurt', 'dmfcc3_kurt', 'dmfcc4_kurt', 'dmfcc5_kurt', 'dmfcc6_kurt', 'dmfcc7_kurt', 'dmfcc8_kurt', 'dmfcc9_kurt', 'dmfcc10_kurt', 'dmfcc11_kurt', 'dmfcc12_kurt', 'dmfcc13_kurt', 'ddmfcc1_kurt', 'ddmfcc2_kurt', 'ddmfcc3_kurt', 'ddmfcc4_kurt', 'ddmfcc5_kurt', 'ddmfcc6_kurt', 'ddmfcc7_kurt', 'ddmfcc8_kurt', 'ddmfcc9_kurt', 'ddmfcc10_kurt', 'ddmfcc11_kurt', 'ddmfcc12_kurt', 'ddmfcc13_kurt']
    feature_cols = acoustic_feats + mfcc_feats + ['y', 'subject_id', 'sample_id', 'gender', 'dataset']
    fimp_sorted = sorted(zip(feature_cols, fimp), key=lambda l: l[1], reverse=True)
    fimp_sorted = [(f0, round(f1, 4)) for f0, f1 in fimp_sorted]

    print([f0 for f0,f1 in fimp_sorted[:20] if f1 > 0.015])

    # [6, 5, 6, 6, 156]
    fl = [0, 6, 11, 17, 23, 179]
    f0_features = [x for x in range(fl[0], fl[1])]
    jitter_features = [x for x in range(fl[1], fl[2])]
    shimmer_features = [x for x in range(fl[2], fl[3])]
    formant_features = [x for x in range(fl[3], fl[4])]
    mfcc_features = [x for x in range(fl[4], fl[5])]

    sum_f0, sum_jit, sum_shim, sum_form, sum_mfcc = 0, 0,0,0,0
    for i, value in enumerate(fimp):
        if i in f0_features:
            sum_f0 += abs(value)
        if i in jitter_features:
            sum_jit += abs(value)
        if i in shimmer_features:
            sum_shim += abs(value)
        if i in formant_features:
            sum_form += abs(value)
        if i in mfcc_features:
            sum_mfcc += abs(value)

    print("Contribution of F0: {:.3f} (avg: {:.3f})".format(sum_f0, sum_f0/len(f0_features)))
    # sum_jit = sum(val for key, val in fimp if key in jitter_features)
    print("Contribution of Jitter: {:.3f} (avg: {:.3f})".format(sum_jit, sum_jit/len(jitter_features)))
    # sum_shim = sum(val for key, val in fimp if key in shimmer_features)
    print("Contribution of Shimmer: {:.3f} (avg: {:.3f})".format(sum_shim, sum_shim/len(shimmer_features)))
    # sum_form = sum(val for key, val in fimp if key in formant_features)
    print("Contribution of formants: {:.3f} (avg: {:.3f})".format(sum_form, sum_form/len(formant_features)))
    # sum_mfcc = sum(val for key, val in fimp if key in mfcc_features)
    print("Contribution of MFCC: {:.3f} (avg: {:.3f})".format(sum_mfcc, sum_mfcc/len(mfcc_features)))
    print("Total feature importance (should equal to 1):", sum_mfcc + sum_f0 + sum_form + sum_jit + sum_shim)

    cmap = plt.cm.get_cmap('tab20b_r')
    colors = [cmap(i) for i in range(20)]
    fig = plt.figure(figsize=(6, 8))
    for i in range(len(fl)-1):
        plt.barh(df.columns[fl[i]:fl[i+1]], fimp[fl[i]:fl[i+1]], color=colors[i*4+2])

    for i in range(4):  # Mean, std, skewness, kurtosis
        for j in range(3): # MFCC and derivatives
            plt.barh(df.columns[fl[4]+i*39+j*13:fl[4]+i*39+j*13+13], fimp[fl[4]+i*39+j*13:fl[4]+i*39+j*13+13], color=colors[16+j])
    # plt.barh(df.columns[fl[4]+39:fl[4]+39+39], fimp[fl[4]+39:fl[4]+39+39], color=colors[17])
    # plt.barh(df.columns[fl[4]+39+39:fl[4]+78+39], fimp[fl[4]+39+39:fl[4]+78+39], color=colors[18])
    # plt.barh(df.columns[fl[4]+78+39:fl[4]+78+78], fimp[fl[4]+78+39:fl[4]+78+78], color=colors[19])

    plt.yticks(
        [(fl[0] + fl[1]) / 2, (fl[1] + fl[2]) / 2, (fl[2] + fl[3]) / 2, (fl[3] + fl[4]) / 2, 
         (fl[5] - fl[4]) / 8 + fl[4], (fl[5] - fl[4]) / 8 * 3 + fl[4], (fl[5] - fl[4]) / 8 * 6, (fl[5] - fl[4]) ],
        ['F0', 'Jitter', 'Shimmer', 'Formants', f'MFCC \n Mean', f'MFCC \n Std. Dev.', f'MFCC \n Skewness', f'MFCC \n Kurtosis'])
    plt.xlabel("Relative feature importance")
    plt.title(f"Feature importance for the IFM model on {' '.join([s for s in df.at[0, 'dataset'].split('_')])} data") 	
    plt.tight_layout()
    plt.ylim((fl[0], fl[-1]))
    fig.savefig(os.path.join(experiment_folder, f'fimp_{df.at[0, "dataset"]}.pdf'), dpi=300)
    plt.show()

def fimp_plot_nifm(fimp, df):
    nifm_features = list(range(df.shape[1]-5))
    feature_cols = nifm_features + ['y', 'subject_id', 'sample_id', 'gender', 'dataset']
    fimp_sorted = sorted(zip(feature_cols, fimp), key=lambda l: l[1], reverse=True)
    fimp_sorted = [(f0, round(f1, 4)) for f0, f1 in fimp_sorted]

    print(sorted([f0 for f0,f1 in fimp_sorted[:20] if f1 > 0.015]))

    plt.barh(df.columns[:len(nifm_features)], fimp, color='green')

    plt.xlabel("Relative feature importance")
    plt.tight_layout()

    # plt.show()