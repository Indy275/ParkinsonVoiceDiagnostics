import random
import numpy as np
import pandas as pd


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
    store_location = 'C:\\Users\\INDYD\\Documents\\RAIVD_data\\preprocessed_data\\{}_preprocessed\\'.format(dataset)

    X = np.load(store_location + 'X_{}.npy'.format(ifm_nifm))
    y = np.load(store_location + 'y_{}.npy'.format(ifm_nifm))
    subj_id = np.load(store_location + 'subj_id_{}.npy'.format(ifm_nifm))
    sample_id = np.load(store_location + 'sample_id_{}.npy'.format(ifm_nifm))
    train_data = np.load(store_location + 'train_data_{}.npy'.format(ifm_nifm))

    # TODO place dataframe creation to feature extraction
    n_features = X.shape[1]
    data = np.hstack((X, y, subj_id, sample_id, train_data))
    df = pd.DataFrame(data=data, columns=[str(x) for x in list(range(n_features))] + ['y', 'subj_id', 'sample_id', 'train_test'])

    return df, n_features


def split_data(X, y, train_data):
    X_train = [x for x, is_train in zip(X, train_data) if is_train]
    X_test = [x for x, is_train in zip(X, train_data) if not is_train]
    y_train = [y for y, is_train in zip(y, train_data) if is_train]
    y_test = [y for y, is_train in zip(y, train_data) if not is_train]

    print("The training data consists of {} samples, of which {} of PD patients ({:.1f}%)".format(len(y_train),
                                                                                                  int(sum(y_train)),
                                                                                                  float(
                                                                                                      sum(y_train) / len(
                                                                                                          y_train) * 100)))
    print("The test data consists of {} samples, of which {} of PD patients ({:.1f}%)".format(len(y_test),
                                                                                              int(sum(y_test)),
                                                                                              float(sum(y_test) / len(
                                                                                                  y_test) * 100)))
    return X_train, X_test, y_train, y_test
