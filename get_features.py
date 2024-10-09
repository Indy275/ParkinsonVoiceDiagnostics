import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import get_ifm_features
import get_nifm_features
from data_util.data_util import make_train_test_split
from data_util.file_util import load_files, get_dirs


def scale_features(df, X):
    # Scale the train and test data, fit to the train data
    scaler = StandardScaler()
    n_features = X.shape[1]

    train_indices = df['train_test'] == 'True'
    test_indices = df['train_test'] == 'False'

    X_train = df.loc[train_indices, df.columns[:n_features]]
    X_test = df.loc[test_indices, df.columns[:n_features]]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    df.loc[train_indices, df.columns[:n_features]] = X_train_scaled
    df.loc[test_indices, df.columns[:n_features]] = X_test_scaled
    return df


def create_features(dataset, ifm_nifm):
    dir, store_location = get_dirs(dataset)

    files, HC_id_list, PD_id_list = load_files(dir)

    HC_train, HC_test = make_train_test_split(HC_id_list)
    PD_train, PD_test = make_train_test_split(PD_id_list)

    print("Found {} speakers, of which {} PD and {} HC.".format(len(HC_id_list) + len(PD_id_list), len(PD_id_list),
                                                                len(HC_id_list)))

    print("Created a train set containing {} PD and {} HC speakers.".format(len(PD_train), len(HC_train)))
    print("Created a test set containing {} PD and {} HC speakers.".format(len(PD_test), len(HC_test)))

    X, y, subj_id, sample_id, train_data = [], [], [], [], []
    id_count = 0

    for file in files:
        path_to_file = os.path.join(dir, file) + '.wav'

        if ifm_nifm[:3] == 'ifm':
            if ifm_nifm[-4:] == 'file':
                static = True
            elif ifm_nifm[-6:] == 'window':
                static = False
            else:
                print("Something went wrong:", ifm_nifm)
                return

            features = get_ifm_features.get_features(path_to_file, static)
        elif ifm_nifm[:4] == 'nifm':
            features = get_nifm_features.get_features(dataset)
        else:
            print("Something went wrong:", ifm_nifm)
            return

        X.extend(features)
        y.extend([1 if file[:2] == 'PD' else 0] * features.shape[0])
        subj_id.extend([file[-4:]] * features.shape[0])
        sample_id.extend([id_count] * features.shape[0])
        train_data.extend([str(file[-4:]) in PD_train + HC_train] * features.shape[0])
        id_count += 1

    X = np.vstack(X)
    y = np.array(y).reshape(-1, 1)
    subj_id = np.array(subj_id).reshape(-1, 1)
    sample_id = np.array(sample_id).reshape(-1, 1)
    train_data = np.array(train_data).reshape(-1, 1)
    data = np.hstack((X, y, subj_id, sample_id, train_data))
    df = pd.DataFrame(data=data, columns=list(range(X.shape[1])) + ['y', 'subject_id', 'sample_id', 'train_test'])
    df = scale_features(df, X)

    print("Identified {} files, of which {} from PWP and {} from HC.".format(len(y),
                                                                             len([i for i in y if
                                                                                  i == 1]),
                                                                             len([i for i in y if
                                                                                  i == 0])))

    df.to_csv(store_location + "data_{}.csv".format(ifm_nifm), index=False)
    print('Data with shape {} saved to {}.'.format(X.shape, store_location))
