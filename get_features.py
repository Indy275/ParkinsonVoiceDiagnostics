import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_util import make_train_test_split
from file_util import load_files, get_dirs


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

def save_intermediate_results(X, y, subj_id, sample_id, gender, train_data, ifm_nifm, store_location, id):
    import get_nifm_features
    X = np.vstack(X)
    y = np.array(y).reshape(-1, 1)
    subj_id = np.array(subj_id).reshape(-1, 1)
    sample_id = np.array(sample_id).reshape(-1, 1)
    gender = np.array(gender).reshape(-1, 1)
    train_data = np.array(train_data).reshape(-1, 1)

    data = np.hstack((X, y, subj_id, sample_id, gender, train_data))
    df = pd.DataFrame(data=data, columns=list(range(X.shape[1])) + ['y', 'subject_id', 'sample_id', 'gender', 'train_test'])
    
    df = scale_features(df, X)
    if ifm_nifm.startswith('nifm'):
        df = get_nifm_features.reduce_dims(df, X.shape[1])


    print("Identified {} files, of which {} from PWP and {} from HC.".format(len(y),
                                                                             len([i for i in y if
                                                                                  i == 1]),
                                                                             len([i for i in y if
                                                                                  i == 0])))

    df.to_csv(os.path.join(store_location, f"data_{ifm_nifm}_{id}.csv"), index=False)
    print(f'Data saved to {store_location}.')

def combine_dfs(store_location, ifm_nifm):
    """
    Combine all partial DataFrames to a full DataFrame. 
    """
    partial_dfs = []
    for file in os.listdir(store_location):
        path_to_file = os.path.join(store_location, file)
        if file.startswith(f'data_{ifm_nifm}_'):
            partial_df = pd.read_csv(path_to_file)
            partial_dfs.append(partial_df)
            os.remove(path_to_file)

    df = pd.concat(partial_dfs)
    df.to_csv(os.path.join(store_location, f"data_{ifm_nifm}.csv"), index=False)



def create_features(dataset, ifm_nifm):
    dir, store_location = get_dirs(dataset)

    files, HC_id_list, PD_id_list = load_files(dir)
    parent_dir = os.path.dirname(dir[:-1])
    genderinfo = pd.read_csv(os.path.join(parent_dir, 'gender.csv'), header=0)

    HC_train, HC_test = make_train_test_split(HC_id_list)
    PD_train, PD_test = make_train_test_split(PD_id_list)

    print("Found {} speakers, of which {} PD and {} HC.".format(len(HC_id_list) + len(PD_id_list), len(PD_id_list),
                                                                len(HC_id_list)))

    print("Created a train set containing {} PD and {} HC speakers.".format(len(PD_train), len(HC_train)))
    print("Created a test set containing {} PD and {} HC speakers.".format(len(PD_test), len(HC_test)))

    X, y, subj_id, sample_id, train_data, gender = [], [], [], [], [], []

    for id, file in enumerate(files):
        print("Processing file {} of {}".format(id, len(files)))
        path_to_file = os.path.join(dir, file) + '.wav'

        if ifm_nifm.startswith('ifm'):
            import get_ifm_features
            features = get_ifm_features.get_features(path_to_file, ifm_nifm)
        elif ifm_nifm.startswith('nifm'):
            import get_nifm_features
            features = get_nifm_features.get_features(path_to_file)
            features = np.squeeze(features)
        else:
            print("Something went wrong:", ifm_nifm)
            return
    
        X.extend(features)
        y.extend([1 if file[:2] == 'PD' else 0] * features.shape[0])
        subj_id.extend([file[-4:]] * features.shape[0])
        sample_id.extend([id] * features.shape[0])
        train_data.extend([str(file[-4:]) in PD_train + HC_train] * features.shape[0])
        gender.extend([genderinfo.loc[genderinfo['ID']==int(file[-4:]), 'Sex'].item()] * features.shape[0])

        if id % 45 == 0 and id > 0:
            save_intermediate_results(X, y, subj_id, sample_id, gender, train_data, ifm_nifm, store_location, id)
            X, y, subj_id, sample_id, train_data, gender = [], [], [], [], [], []  # Start with fresh variables
        if id == len(files)-1:
            save_intermediate_results(X, y, subj_id, sample_id, gender, train_data, ifm_nifm, store_location, id)

    X = np.vstack(X)
    y = np.array(y).reshape(-1, 1)
    subj_id = np.array(subj_id).reshape(-1, 1)
    sample_id = np.array(sample_id).reshape(-1, 1)
    gender = np.array(gender).reshape(-1, 1)
    train_data = np.array(train_data).reshape(-1, 1)

    data = np.hstack((X, y, subj_id, sample_id, gender, train_data))
    df = pd.DataFrame(data=data, columns=list(range(X.shape[1])) + ['y', 'subject_id', 'sample_id', 'gender', 'train_test'])
    
    df = scale_features(df, X)
    if ifm_nifm.startswith('nifm'):
        df = get_nifm_features.reduce_dims(df, X.shape[1])


    print("Identified {} files, of which {} from PWP and {} from HC.".format(len(y),
                                                                             len([i for i in y if
                                                                                  i == 1]),
                                                                             len([i for i in y if
                                                                                  i == 0])))

    df.to_csv(os.path.join(store_location, f"data_{ifm_nifm}.csv"), index=False)

    combine_dfs(store_location, ifm_nifm)
    print(f'Data saved to {store_location}.')
