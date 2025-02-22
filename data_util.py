import random
import pandas as pd
import os
import configparser
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

if os.getenv("COLAB_RELEASE_TAG"):  # colab
    data_dir = '/content/drive/My Drive/RAIVD_data/'
elif os.name == 'posix':  # linux
    data_dir = '/home/indy/Documents/RAIVD_data/'
elif os.name == 'nt':  # windows
    data_dir = "C:\\Users\INDYD\Documents\RAIVD_data\\"

config = configparser.ConfigParser()
config.read('settings.ini')
normalize_audio = config.getboolean('DATA_SETTINGS', 'normalize_audio')


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
    norm = '_norm' if normalize_audio else ''
    df = pd.read_csv(os.path.join(data_dir, 'preprocessed_data',f'{dataset}{norm}_{ifm_nifm}.csv'), header=0)
    n_features = len(df.columns) - 5  # Ugly coding, but does the trick: all columns except last 4 are features
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

    return scaler, df


def get_samples(seed, pos_subjs, neg_subjs, n_shots, df):
    random.seed(seed)
    pos_train_samples = random.sample(pos_subjs, min(n_shots, len(pos_subjs)))
    random.seed(seed)
    neg_train_samples = random.sample(neg_subjs, min(n_shots, len(neg_subjs)))

    train_df = df[df['subject_id'].isin(pos_train_samples + neg_train_samples)]
    test_df = df[~df['subject_id'].isin(pos_train_samples + neg_train_samples)]

    # print("train subjects:", list(np.unique(train_df['subject_id'])))
    # print("test subjects:", list(np.unique(test_df['subject_id'])))

    return train_df, test_df
