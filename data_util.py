import random
import numpy as np
import pandas as pd
import configparser
import os

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
    store_location = os.path.join(data_dir, 'preprocessed_data', f'{dataset}_preprocessed')

    df = pd.read_csv(os.path.join(store_location,f'data_{ifm_nifm}.csv'))
    n_features = len(df.columns) - 5  # Ugly coding, but does the trick: all columns except last 5 are features
    return df, n_features
