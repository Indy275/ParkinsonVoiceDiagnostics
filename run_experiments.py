import numpy as np
import configparser
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import pandas as pd
import os
from copy import deepcopy

from data_util import load_data, scale_features
from DNN_models import run_dnn_model
from ML_models import run_ml_model

config = configparser.ConfigParser()
config.read('settings.ini')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')
pd.options.mode.chained_assignment = None  # default='warn'

cwd = os.path.abspath(os.getcwd())
experiment_folder = os.path.join(cwd,'experiments')
if not os.path.exists(experiment_folder):
    os.makedirs(experiment_folder)


def run_data_fold(model, df, n_features, train_indices, test_indices):
    X_train = df.loc[train_indices, df.columns[:n_features]]
    X_test = df.loc[test_indices, df.columns[:n_features]]
    y_train = df.loc[train_indices, 'y']
    y_test = df.loc[test_indices, 'y']

    if print_intermediate:
        print("Train subjects:", np.sort(df.loc[train_indices, 'subject_id'].unique()))
        print("Test subjects:", np.sort(df.loc[test_indices, 'subject_id'].unique()))
        print("Train/test shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    if model == 'RFC':
        return run_ml_model(X_train, X_test, y_train, y_test, df, test_indices)
    elif model == 'DNN':
        import tensorflow as tf
        X_train = tf.convert_to_tensor(X_train)
        X_test = tf.convert_to_tensor(X_test)
        y_train = tf.convert_to_tensor(y_train)
        y_test = tf.convert_to_tensor(y_test)

        return run_dnn_model(X_train, X_test, y_train, y_test, df, test_indices)


def run(dataset, ifm_nifm, model, k=1):
    df, n_features = load_data(dataset, ifm_nifm)
    print("Data shape:", df.shape)

    fscores, sscores = [], []
    split_df = df.drop_duplicates(['subject_id'])
    split_df.loc[:,'ygender'] = split_df['y'].astype(str) + '_'# + split_df['gender'].astype(str)
    print(split_df['ygender'].value_counts())

    kf = StratifiedKFold(n_splits=k, shuffle=True)
    
    for i, (train_split_indices, test_split_indices) in enumerate(kf.split(split_df['subject_id'], split_df['ygender'])):
        print(f"Running {model} with data fold {i} of {k}")
        df = deepcopy(df)

        # Crazy hack needed because split() function stupidity... only works if sample_id is unique
        trainsamples = split_df.iloc[train_split_indices]['sample_id']
        testsamples = split_df.iloc[test_split_indices]['sample_id']
        train_indices = df[df['sample_id'].isin(trainsamples)].index.tolist()
        test_indices = df[df['sample_id'].isin(testsamples)].index.tolist()

        print(train_split_indices, test_split_indices)
        print(df.iloc[test_split_indices])
    
        df = scale_features(df, n_features, train_indices, test_indices)

        fscore, sscore = run_data_fold(model, df, n_features, train_indices, test_indices)
        fscores.append(fscore)
        sscores.append(sscore)

    print("File-level performance:")
    print("Mean Acc:", round(sum(i[0] for i in fscores)/k, 3))
    print("Mean AUC:", round(sum(i[1] for i in fscores)/k, 3))
    print("Mean Sens:", round(sum(i[2] for i in fscores)/k, 3))
    print("Mean Spec:", round(sum(i[3] for i in fscores)/k, 3))

    print("Speaker-level performance:")
    print("Mean Acc:", round(sum(i[0] for i in sscores)/k, 3))
    print("Mean AUC:", round(sum(i[1] for i in sscores)/k, 3))
    print("Mean Sens:", round(sum(i[2] for i in sscores)/k, 3))
    print("Mean Spec:", round(sum(i[3] for i in sscores)/k, 3))



