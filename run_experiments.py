import numpy as np
import configparser
from sklearn.model_selection import train_test_split
import pandas as pd
import os

from data_util import load_data
from DNN_models import run_dnn_model
from ML_models import run_ml_model

config = configparser.ConfigParser()
config.read('settings.ini')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')
pd.options.mode.chained_assignment = None  # default='warn'


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
    cwd = os.path.abspath(os.getcwd())
    experiment_folder = os.path.join(cwd,'experiments')
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    df, n_features = load_data(dataset, ifm_nifm)
    print("Data shape:", df.shape)

    fscores, sscores = [], []
    if k == 1:
        df['train_test'] = df['train_test'].astype(bool)
        train_indices = df['train_test']
        test_indices = ~df['train_test']
        fscore, sscore = run_data_fold(model, df, n_features, train_indices, test_indices)
        fscores.append(fscore)
        sscores.append(sscore)
    else:
        for i in range(k):
            print(f"Running {model} with data fold {i} of {k}")
            split_df = df.drop_duplicates(['subject_id'])
            split_df.loc[:,'ygender'] = split_df['y'].astype(str) + '_' + split_df['gender'].astype(str)
            train_subjects, test_subjects = train_test_split(split_df['subject_id'], shuffle=True, stratify=split_df['ygender'])

            train_indices = df.index[df['subject_id'].isin(train_subjects)].tolist()
            test_indices = df.index[df['subject_id'].isin(test_subjects)].tolist()

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



