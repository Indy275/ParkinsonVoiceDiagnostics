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

    file_metrics, subject_metrics = [], []
    split_df = df.drop_duplicates(['subject_id'])
    split_df.loc[:,'ygender'] = split_df['y'].astype(str) + '_' + split_df['gender'].astype(str)

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
        print("Mean Acc:", round(np.mean(subject_metrics)[0], 3))
        print("Mean AUC:", round(np.mean(np.mean(subject_metrics))[1], 3))
        print("Mean Sens:", round(sum(i[2] for i in subject_metrics)/k, 3))
        print("Mean Spec:", round(sum(i[3] for i in subject_metrics)/k, 3))
        
        print("Base data performance:")
        print("Mean Acc:", round(sum(i[0] for i in subject_metrics)/k, 3))
        print("Mean AUC:", round(sum(i[1] for i in subject_metrics)/k, 3))
        print("Mean Sens:", round(sum(i[2] for i in subject_metrics)/k, 3))
        print("Mean Spec:", round(sum(i[3] for i in subject_metrics)/k, 3))
