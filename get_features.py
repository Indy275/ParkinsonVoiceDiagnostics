import os.path

import numpy as np
import pandas as pd

from file_util import load_files, get_dirs


def save_intermediate_results(X, y, subj_id, sample_id, gender, ifm_nifm, store_location, id):
    X = np.vstack(X)
    y = np.array(y).reshape(-1, 1)
    subj_id = np.array(subj_id).reshape(-1, 1)
    sample_id = np.array(sample_id).reshape(-1, 1)
    gender = np.array(gender).reshape(-1, 1)

    data = np.hstack((X, y, subj_id, sample_id, gender))
    df = pd.DataFrame(data=data, columns=list(range(X.shape[1])) + ['y', 'subject_id', 'sample_id', 'gender'])
    if ifm_nifm.startswith('nifm'):
        import get_nifm_features
        df = get_nifm_features.reduce_dims(df, len(df.columns)-4)

    df.to_csv(os.path.join(store_location[0], f"{store_location[1]}_{ifm_nifm}_{id}.csv"), index=False)
    print(f'Intermediate data saved to {store_location[0]}.')

def combine_dfs(store_location, ifm_nifm):
    """
    Combine all partial DataFrames to a full DataFrame. 
    """

    partial_dfs = []
    for file in os.listdir(store_location[0]):
        path_to_file = os.path.join(store_location[0], file)
        if file.startswith(f'{store_location[1]}_{ifm_nifm}_'):
            partial_df = pd.read_csv(path_to_file)
            partial_dfs.append(partial_df)
            os.remove(path_to_file)

    df = pd.concat(partial_dfs)
        
    print("Identified {} files, of which {} from PWP and {} from HC.".format(df.shape[0],
                                                                             len([i for i in df['y'] if
                                                                                  i == 1]),
                                                                             len([i for i in df['y'] if
                                                                                  i == 0])))
    
    # if ifm_nifm.startswith('nifm'):
    #     import get_nifm_features
    #     # df = get_nifm_features.reduce_dims(df, len(df.columns)-5)
    #     df = get_nifm_features.aggregate_windows(df)

    df.to_csv(os.path.join(store_location[0], f"{store_location[1]}_{ifm_nifm}.csv"), index=False)



def create_features(dataset, ifm_nifm):
    dir, store_location = get_dirs(dataset)

    files, HC_id_list, PD_id_list = load_files(dir)
    parent_dir = os.path.dirname(dir[:-1])
    genderinfo = pd.read_csv(os.path.join(parent_dir, 'gender.csv'), header=0)

    print("Found {} speakers, of which {} PD and {} HC.".format(len(HC_id_list) + len(PD_id_list), len(PD_id_list),
                                                                len(HC_id_list)))

    X, y, subj_id, sample_id, gender = [], [], [], [], []

    for id, file in enumerate(files):
        print("Processing file {} of {}".format(id, len(files)))
        path_to_file = os.path.join(dir, file) + '.wav'

        if ifm_nifm.startswith('ifm'):
            import get_ifm_features
            features = get_ifm_features.get_features(path_to_file)
        elif ifm_nifm.startswith('nifm'):
            import get_nifm_features
            features = get_nifm_features.get_features(path_to_file)
    
        X.extend(features)
        y.extend([1 if file[:2] == 'PD' else 0] * features.shape[0])
        subj_id.extend([file[-4:]] * features.shape[0])
        sample_id.extend([id] * features.shape[0])
        gender.extend([genderinfo.loc[genderinfo['ID']==int(file[-4:]), 'Sex'].item()] * features.shape[0])
        # updrs.extend([genderinfo.loc[genderinfo['ID']==int(file[-4:]), 'UPDRS scale'].item()] * features.shape[0])

        if id % 20 == 0 and id > 0:
            save_intermediate_results(X, y, subj_id, sample_id, gender, ifm_nifm, store_location, id)
            X, y, subj_id, sample_id, gender = [], [], [], [], []  # Start with fresh variables
        if id == len(files)-1:
            save_intermediate_results(X, y, subj_id, sample_id, gender, ifm_nifm, store_location, id)

    combine_dfs(store_location, ifm_nifm)
    print(f'Data saved to {store_location}.')
