import random
import numpy as np
import pandas as pd
import configparser
import os

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'settings.ini'))
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
    store_location = data_dir + 'preprocessed_data\\{}_preprocessed\\'.format(dataset)

    df = pd.read_csv(store_location+'data_{}.csv'.format(ifm_nifm))
    n_features = len(df.columns) - 5  # Ugly coding, but does the trick: all columns except last 5 are features
    return df, n_features


# def split_data(X, y, train_data):
#     X_train = [x for x, is_train in zip(X, train_data) if is_train]
#     X_test = [x for x, is_train in zip(X, train_data) if not is_train]
#     y_train = [y for y, is_train in zip(y, train_data) if is_train]
#     y_test = [y for y, is_train in zip(y, train_data) if not is_train]

#     print("The training data consists of {} samples, of which {} of PD patients ({:.1f}%)".format(len(y_train),
#                                                                                                   int(sum(y_train)),
#                                                                                                   float(
#                                                                                                       sum(y_train) / len(
#                                                                                                           y_train) * 100)))
#     print("The test data consists of {} samples, of which {} of PD patients ({:.1f}%)".format(len(y_test),
#                                                                                               int(sum(y_test)),
#                                                                                               float(sum(y_test) / len(
#                                                                                                   y_test) * 100)))
#     return X_train, X_test, y_train, y_test
