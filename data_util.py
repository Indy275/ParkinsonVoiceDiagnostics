import random
import numpy as np
import pandas as pd
from joblib import dump
import configparser
import os
from sklearn.preprocessing import StandardScaler
import torch

config = configparser.ConfigParser()
config.read('settings.ini')
data_dir = config['DEFAULT']['data_dir']


def make_train_test_split(id_list, test_size=0.3, seed=1):
    """
    Divide a list into a training set and a testing set according to a given test set percentage
    """
    random.Random(seed).shuffle(id_list)
    cut = int(test_size * len(id_list))
    train_set = id_list[cut:]
    test_set = id_list[:cut]
    return train_set, test_set


def load_data(dataset, ifm_nifm):
    df = pd.read_csv(os.path.join(data_dir, 'preprocessed_data',f'{dataset}_{ifm_nifm}.csv'), header=0)
    n_features = len(df.columns) - 4  # Ugly coding, but does the trick: all columns except last 4 are features
    return df, n_features


def scale_features(df, n_features, train_indices, test_indices):
    scaler = StandardScaler()

    X_train = df.loc[train_indices, df.columns[:n_features]]
    X_test = df.loc[test_indices, df.columns[:n_features]]

    scaler.partial_fit(X_train.values)
    X_train_scaled = scaler.transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    df.loc[train_indices, df.columns[:n_features]] = X_train_scaled
    df.loc[test_indices, df.columns[:n_features]] = X_test_scaled

    # dump(scaler, 'scaler.pkl')
    
    return scaler, df


def get_samples(seed, pos_subjs, neg_subjs, n_shots, df):
    random.seed(seed)
    pos_train_samples = random.sample(pos_subjs, min(n_shots, len(pos_subjs)))
    random.seed(seed)
    neg_train_samples = random.sample(neg_subjs, min(n_shots, len(neg_subjs)))

    train_df = df[df['subject_id'].isin(pos_train_samples + neg_train_samples)]
    test_df = df[~df['subject_id'].isin(pos_train_samples + neg_train_samples)]

    return train_df, test_df
