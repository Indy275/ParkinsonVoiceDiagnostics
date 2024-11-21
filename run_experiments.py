import os
from copy import deepcopy
import numpy as np
import pandas as pd
import configparser

import librosa
import librosa.display as display
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from data_util import load_data, scale_features

config = configparser.ConfigParser()
config.read('settings.ini')
plot_fimp = config.getboolean('OUTPUT_SETTINGS', 'plot_fimp')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')
clf = config['MODEL_SETTINGS']['clf']
gender = config.getint('EXPERIMENT_SETTINGS', 'gender')


pd.options.mode.chained_assignment = None  # default='warn'
if clf.startswith('DNN'):
    from DNN_models import run_dnn_model, run_dnn_tl_model, run_dnn_fstl_model
elif clf.startswith('SVM'):
    from ML_models import run_ml_model, run_ml_tl_model, run_ml_fstl_model
elif clf == 'PCA_PLDA':
    from PCA_PLDA import run_PCA_PLDA


cwd = os.path.abspath(os.getcwd())
experiment_folder = os.path.join(cwd,'experiments')
if not os.path.exists(experiment_folder):
    os.makedirs(experiment_folder)


def run_data_fold(model, df, train_indices, test_indices):
    train_df = df.loc[train_indices, :]
    test_df = df.loc[test_indices, :]

    if print_intermediate:
        print("Train subjects:", np.sort(train_df.loc[:, 'subject_id'].unique()), '({})'.format(len(np.sort(train_df.loc[:, 'subject_id'].unique()))))
        print("Test subjects:", np.sort(df.loc[:, 'subject_id'].unique()), '({})'.format(len(np.sort(test_df.loc[:, 'subject_id'].unique()))))
        # print("Train/test shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        print("Train %PD:",round(train_df.loc[:, 'y'].sum() / len(train_indices),3))
        print("Test %PD:",round(test_df.loc[:, 'y'].sum() / len(test_indices),3))
        print("Train %male",round(train_df.loc[:, 'gender'].sum() / len(train_indices),3))
        print("Test %male",round(test_df.loc[:, 'gender'].sum()/  len(test_indices),3))

    if model.startswith('SVM'):
        return run_ml_model(train_df, test_df)
    elif model == 'PCA_PLDA':
        print("Model PCA-PLDA is no longer supported")
        # return run_PCA_PLDA(train_df, test_df)
    elif model.startswith('DNN'):
        return run_dnn_model(model, train_df, test_df)


def run_monolingual(dataset, ifm_nifm, model, k=2):
    df, n_features = load_data(dataset, ifm_nifm)

    # Experiment: only include Male/Female participants
    if gender < 2:
        df = df[df['gender']==gender]

    file_metrics, subject_metrics, fimps = [], [], []
    split_df = df.drop_duplicates(['subject_id'])
    split_df.loc[:,'ygender'] = split_df['y'].astype(str) #+ '_' + split_df['gender'].astype(str)

    print(f"Data loaded succesfully with shapes {df.shape}, now running {model} classifier")
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    for i, (train_split_indices, test_split_indices) in enumerate(kf.split(split_df['subject_id'], split_df['ygender'])):
        print(f"Data fold [{i+1}/{k}]")
        df_copy = deepcopy(df)

        train_subjects = split_df.iloc[train_split_indices]['subject_id']
        test_subjects = split_df.iloc[test_split_indices]['subject_id']
        train_indices = df_copy[df_copy['subject_id'].isin(train_subjects)].index.tolist()
        test_indices = df_copy[df_copy['subject_id'].isin(test_subjects)].index.tolist()

        _, df_copy = scale_features(df_copy, n_features, train_indices, test_indices)
        if plot_fimp:
            file_metric, subject_metric, fimp = run_data_fold(model, df_copy, train_indices, test_indices)
            fimps.append(fimp)
        else: 
            file_metric, subject_metric = run_data_fold(model, df_copy, train_indices, test_indices)

        file_metrics.append(file_metric)
        subject_metrics.append(subject_metric)
    print(f"Average {k}-fold performance of {model}-{ifm_nifm} model with {dataset} data:")
    if dataset[-3:] != 'tdu' and dataset[-3:] != 'ddk':
        print("File-level performance:")
        print("Mean Acc:", round(sum(i[0] for i in file_metrics)/k, 3))
        print("Mean AUC:", round(sum(i[1] for i in file_metrics)/k, 3))
        print("Mean Sens:", round(sum(i[2] for i in file_metrics)/k, 3))
        print("Mean Spec:", round(sum(i[3] for i in file_metrics)/k, 3))

    print("Speaker-level performance:")
    print("Mean Acc:", round(sum(i[0] for i in subject_metrics)/k, 3))
    print("Mean AUC:", round(sum(i[1] for i in subject_metrics)/k, 3))
    print("Mean Sens:", round(sum(i[2] for i in subject_metrics)/k, 3))
    print("Mean Spec:", round(sum(i[3] for i in subject_metrics)/k, 3))

    if plot_fimp:
        fimp = np.mean(fimps, axis=0)
        acoustic_feats = ['F0_mean', 'F0_std','F0_min', 'F0_max', 'dF0_mean', 'ddF0_mean', '%Jitter', 'absJitter', 'RAP', 'PPQ5', 'DDP', '%Shimmer', 'dbShimmer', 'APQ3', 'APQ5', 'APQ11', 'DDA','F1_mean','F2_mean','F3_mean']
        mfcc_feats = ['mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean', 'mfcc5_mean', 'mfcc6_mean', 'mfcc7_mean', 'mfcc8_mean', 'mfcc9_mean', 'mfcc10_mean', 'mfcc11_mean', 'mfcc12_mean', 'mfcc13_mean', 'dmfcc1_mean', 'dmfcc2_mean', 'dmfcc3_mean', 'dmfcc4_mean', 'dmfcc5_mean', 'dmfcc6_mean', 'dmfcc7_mean', 'dmfcc8_mean', 'dmfcc9_mean', 'dmfcc10_mean', 'dmfcc11_mean', 'dmfcc12_mean', 'dmfcc13_mean', 'ddmfcc1_mean', 'ddmfcc2_mean', 'ddmfcc3_mean', 'ddmfcc4_mean', 'ddmfcc5_mean', 'ddmfcc6_mean', 'ddmfcc7_mean', 'ddmfcc8_mean', 'ddmfcc9_mean', 'ddmfcc10_mean', 'ddmfcc11_mean', 'ddmfcc12_mean', 'ddmfcc13_mean', 'mfcc1_std', 'mfcc2_std', 'mfcc3_std', 'mfcc4_std', 'mfcc5_std', 'mfcc6_std', 'mfcc7_std', 'mfcc8_std', 'mfcc9_std', 'mfcc10_std', 'mfcc11_std', 'mfcc12_std', 'mfcc13_std', 'dmfcc1_std', 'dmfcc2_std', 'dmfcc3_std', 'dmfcc4_std', 'dmfcc5_std', 'dmfcc6_std', 'dmfcc7_std', 'dmfcc8_std', 'dmfcc9_std', 'dmfcc10_std', 'dmfcc11_std', 'dmfcc12_std', 'dmfcc13_std', 'ddmfcc1_std', 'ddmfcc2_std', 'ddmfcc3_std', 'ddmfcc4_std', 'ddmfcc5_std', 'ddmfcc6_std', 'ddmfcc7_std', 'ddmfcc8_std', 'ddmfcc9_std', 'ddmfcc10_std', 'ddmfcc11_std', 'ddmfcc12_std', 'ddmfcc13_std', 'mfcc1_skew', 'mfcc2_skew', 'mfcc3_skew', 'mfcc4_skew', 'mfcc5_skew', 'mfcc6_skew', 'mfcc7_skew', 'mfcc8_skew', 'mfcc9_skew', 'mfcc10_skew', 'mfcc11_skew', 'mfcc12_skew', 'mfcc13_skew', 'dmfcc1_skew', 'dmfcc2_skew', 'dmfcc3_skew', 'dmfcc4_skew', 'dmfcc5_skew', 'dmfcc6_skew', 'dmfcc7_skew', 'dmfcc8_skew', 'dmfcc9_skew', 'dmfcc10_skew', 'dmfcc11_skew', 'dmfcc12_skew', 'dmfcc13_skew', 'ddmfcc1_skew', 'ddmfcc2_skew', 'ddmfcc3_skew', 'ddmfcc4_skew', 'ddmfcc5_skew', 'ddmfcc6_skew', 'ddmfcc7_skew', 'ddmfcc8_skew', 'ddmfcc9_skew', 'ddmfcc10_skew', 'ddmfcc11_skew', 'ddmfcc12_skew', 'ddmfcc13_skew', 'mfcc1_kurt', 'mfcc2_kurt', 'mfcc3_kurt', 'mfcc4_kurt', 'mfcc5_kurt', 'mfcc6_kurt', 'mfcc7_kurt', 'mfcc8_kurt', 'mfcc9_kurt', 'mfcc10_kurt', 'mfcc11_kurt', 'mfcc12_kurt', 'mfcc13_kurt', 'dmfcc1_kurt', 'dmfcc2_kurt', 'dmfcc3_kurt', 'dmfcc4_kurt', 'dmfcc5_kurt', 'dmfcc6_kurt', 'dmfcc7_kurt', 'dmfcc8_kurt', 'dmfcc9_kurt', 'dmfcc10_kurt', 'dmfcc11_kurt', 'dmfcc12_kurt', 'dmfcc13_kurt', 'ddmfcc1_kurt', 'ddmfcc2_kurt', 'ddmfcc3_kurt', 'ddmfcc4_kurt', 'ddmfcc5_kurt', 'ddmfcc6_kurt', 'ddmfcc7_kurt', 'ddmfcc8_kurt', 'ddmfcc9_kurt', 'ddmfcc10_kurt', 'ddmfcc11_kurt', 'ddmfcc12_kurt', 'ddmfcc13_kurt']
        feature_cols = acoustic_feats + mfcc_feats + ['y', 'subject_id', 'sample_id', 'gender']
        fimp_sorted = sorted(zip(feature_cols, fimp), key=lambda l: l[1], reverse=True)
        print(fimp_sorted[:20])

        # [6, 5, 6, 3, 156]
        fl = [0, 6, 11, 17, 20, 176]
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

        plt.barh(df.columns[fl[0]:fl[1]], fimp[fl[0]:fl[1]], color='green')
        plt.barh(df.columns[fl[1]:fl[2]], fimp[fl[1]:fl[2]], color='blue')
        plt.barh(df.columns[fl[2]:fl[3]], fimp[fl[2]:fl[3]], color='red')
        plt.barh(df.columns[fl[3]:fl[4]], fimp[fl[3]:fl[4]], color='purple')
        # plt.barh(df.columns[fl[4]:fl[5]], fimp[fl[4]:fl[5]], color='orange')

        plt.barh(df.columns[fl[4]:fl[4]+39], fimp[fl[4]:fl[4]+39], color='green')
        plt.barh(df.columns[fl[4]+39:fl[4]+39+39], fimp[fl[4]+39:fl[4]+39+39], color='blue')
        plt.barh(df.columns[fl[4]+39+39:fl[4]+78+39], fimp[fl[4]+39+39:fl[4]+78+39], color='red')
        plt.barh(df.columns[fl[4]+78+39:fl[4]+78+78], fimp[fl[4]+78+39:fl[4]+78+78], color='orange')

        plt.yticks(
            [(fl[0] + fl[1]) / 2, (fl[1] + fl[2]) / 2, (fl[2] + fl[3]) / 2, (fl[3] + fl[4]) / 2, (fl[4] + fl[5]) / 2],
            ['F0', 'Jitter', 'Shimmer', 'Formants', 'MFCC'])
        plt.xlabel("Relative feature importance")
        plt.tight_layout()
        plt.ylim((fl[0], fl[-1]))
        plt.show()


def run_data_fold_tl(scaler, model, base_df, base_train_idc, base_test_idc, tgt_df):
    base_train_df = base_df.loc[base_train_idc, :]
    base_test_df = base_df.loc[base_test_idc, :]

    if print_intermediate:
        print("Train subjects:", np.sort(base_df.loc[base_train_idc, 'subject_id'].unique()))
        print("Test subjects:", np.sort(base_df.loc[base_test_idc, 'subject_id'].unique()))
        # print("Train/test shapes:", base_X_train.shape, base_X_test.shape, base_y_train.shape, base_y_test.shape)
        print("Train %PD:",round(base_df.loc[base_train_idc, 'y'].sum()/ len(base_train_idc),3))
        print("Test %PD:",round(base_df.loc[base_test_idc, 'y'].sum()/ len(base_test_idc),3))
        print("Train %male",round(base_df.loc[base_train_idc, 'gender'].sum()/ len(base_train_idc),3))
        print("Test %male",round(base_df.loc[base_test_idc, 'gender'].sum()/ len(base_test_idc),3)) 

    if model == 'SVM':
        return run_ml_tl_model(scaler, base_train_df, base_test_df, tgt_df)
    if model == 'SVMFSTL':
        return run_ml_fstl_model(scaler, base_train_df, base_test_df, tgt_df)
    elif model.endswith('FSTL'): 
        return run_dnn_fstl_model(scaler, model, base_train_df, base_test_df, tgt_df)
    elif model.startswith('DNN'):
        return run_dnn_tl_model(scaler, model, base_train_df, base_test_df, tgt_df)
    

def run_crosslingual(base_dataset, target_dataset, ifm_nifm, model, k=2):
    base_df, base_features = load_data(base_dataset, ifm_nifm)
    target_df, target_features = load_data(target_dataset, ifm_nifm)
    assert base_features == target_features, "Number of features across languages should be equal: {} and {}".format(
        base_features, target_features)
    base_df['sample_id'] = 'base_' + base_df['sample_id'].astype(str)
    target_df['sample_id'] = 'tgt_' + target_df['sample_id'].astype(str)

    # Experiment: only include Male/Female participants
    if gender < 2:
        base_df = base_df[base_df['gender']==gender]
        target_df = target_df[target_df['gender']==gender]

    file_metrics, subject_metrics, base_metrics = [], [], []
    base_df_split = base_df.drop_duplicates(['subject_id'])
    base_df_split.loc[:,'ygender'] = base_df_split['y'].astype(str) + '_' + base_df_split['gender'].astype(str)

    print(f"Data loaded succesfully with shapes {base_df.shape}, {target_df.shape}, now running {model} classifier")
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    for i, (train_split_indices, test_split_indices) in enumerate(kf.split(base_df_split['subject_id'], base_df_split['ygender'])):
        print(f"Running model with data fold [{i+1}/{k}]")
        base_df_copy = deepcopy(base_df)
        target_df_copy = deepcopy(target_df)

        train_subjects = base_df_split.iloc[train_split_indices]['subject_id']
        test_subjects = base_df_split.iloc[test_split_indices]['subject_id']
        train_indices = base_df_copy[base_df_copy['subject_id'].isin(train_subjects)].index.tolist()
        test_indices = base_df_copy[base_df_copy['subject_id'].isin(test_subjects)].index.tolist()

        scaler, base_df_copy = scale_features(base_df_copy, base_features, train_indices, test_indices)

        metrics = run_data_fold_tl(scaler, model, base_df_copy, train_indices, test_indices, target_df_copy)
        if model.endswith('FSTL'):
            file_metric, subject_metric, base_metric, n_tgt_train_samples = zip(*metrics)
        else:
            file_metric, subject_metric, base_metric = zip(*metrics)
        print(f"Average result for data fold [{i+1}/{k}]:\nFile metrics:",np.mean(file_metric, axis=0))
        print("Subject metrics:",np.mean(subject_metric, axis=0),"\nBase metrics:",np.mean(base_metric, axis=0))

        file_metrics.append(file_metric)
        subject_metrics.append(subject_metric)
        base_metrics.append(base_metric)
    if model.endswith('FSTL'):
        fmetrics_df = pd.DataFrame(np.mean(file_metrics, axis=0), columns=['Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity'])
        fmetrics_df['Iteration'] = n_tgt_train_samples
        fmetrics_df.to_csv(os.path.join('experiments', f'{model}_{ifm_nifm}_metrics_{base_dataset}_{target_dataset}.csv'), index=False)

        smetrics_df = pd.DataFrame(np.mean(subject_metrics,axis=0), columns=['Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity'])
        smetrics_df['Iteration'] = n_tgt_train_samples
        smetrics_df.to_csv(os.path.join('experiments', f'{model}_{ifm_nifm}_metrics_{base_dataset}_{target_dataset}_grouped.csv'), index=False)
        
        base_metrics_df = pd.DataFrame(np.mean(base_metrics,axis=0), columns=['Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity'])
        base_metrics_df['Iteration'] = n_tgt_train_samples
        base_metrics_df.to_csv(os.path.join('experiments', f'{model}_{ifm_nifm}_metrics_{base_dataset}_{target_dataset}_base.csv'), index=False)
        
        print(f'Metrics saved to: experiments/{model}_{ifm_nifm}_metrics_{base_dataset}_{target_dataset}.csv')
    else:  # No Few-Shot
        if base_dataset[-3:] != 'tdu' and base_dataset[-3:] != 'ddk':
            print("Target data (file-level) performance:")
            print("Mean Acc:", round(sum(i[0] for i in file_metrics)/k, 3))
            print("Mean AUC:", round(sum(i[1] for i in file_metrics)/k, 3))
            print("Mean Sens:", round(sum(i[2] for i in file_metrics)/k, 3))
            print("Mean Spec:", round(sum(i[3] for i in file_metrics)/k, 3))
        print(np.shape(file_metrics))

        print("Target data (speaker-level) performance:")
        print("Mean Acc:",  round(np.mean(subject_metrics, axis=0).flatten()[0], 3))
        print("Mean AUC:",  round(np.mean(subject_metrics, axis=0).flatten()[1], 3))
        print("Mean Sens:", round(np.mean(subject_metrics, axis=0).flatten()[2], 3))
        print("Mean Spec:", round(np.mean(subject_metrics, axis=0).flatten()[3], 3))
        
        print("Base data performance:")
        print("Mean Acc:", round(np.mean(base_metrics, axis=0).flatten()[0], 3))
        print("Mean AUC:", round(np.mean(base_metrics, axis=0).flatten()[1], 3))
        print("Mean Sens:", round(np.mean(base_metrics, axis=0).flatten()[2], 3))
        print("Mean Spec:", round(np.mean(base_metrics, axis=0).flatten()[3], 3))
