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
    if ifm_nifm.startswith('ifm'):
        acoustic_feats = ['F0_mean', 'F0_std','F0_min', 'F0_max', 'dF0_mean', 'ddF0_mean', '%Jitter', 'absJitter', 'RAP', 'PPQ5', 'DDP', '%Shimmer', 'dbShimmer', 'APQ3', 'APQ5', 'APQ11', 'DDA','F1_mean','F2_mean','F3_mean']
        mfcc_feats = ['mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean', 'mfcc5_mean', 'mfcc6_mean', 'mfcc7_mean', 'mfcc8_mean', 'mfcc9_mean', 'mfcc10_mean', 'mfcc11_mean', 'mfcc12_mean', 'mfcc13_mean', 'dmfcc1_mean', 'dmfcc2_mean', 'dmfcc3_mean', 'dmfcc4_mean', 'dmfcc5_mean', 'dmfcc6_mean', 'dmfcc7_mean', 'dmfcc8_mean', 'dmfcc9_mean', 'dmfcc10_mean', 'dmfcc11_mean', 'dmfcc12_mean', 'dmfcc13_mean', 'ddmfcc1_mean', 'ddmfcc2_mean', 'ddmfcc3_mean', 'ddmfcc4_mean', 'ddmfcc5_mean', 'ddmfcc6_mean', 'ddmfcc7_mean', 'ddmfcc8_mean', 'ddmfcc9_mean', 'ddmfcc10_mean', 'ddmfcc11_mean', 'ddmfcc12_mean', 'ddmfcc13_mean', 'mfcc1_std', 'mfcc2_std', 'mfcc3_std', 'mfcc4_std', 'mfcc5_std', 'mfcc6_std', 'mfcc7_std', 'mfcc8_std', 'mfcc9_std', 'mfcc10_std', 'mfcc11_std', 'mfcc12_std', 'mfcc13_std', 'dmfcc1_std', 'dmfcc2_std', 'dmfcc3_std', 'dmfcc4_std', 'dmfcc5_std', 'dmfcc6_std', 'dmfcc7_std', 'dmfcc8_std', 'dmfcc9_std', 'dmfcc10_std', 'dmfcc11_std', 'dmfcc12_std', 'dmfcc13_std', 'ddmfcc1_std', 'ddmfcc2_std', 'ddmfcc3_std', 'ddmfcc4_std', 'ddmfcc5_std', 'ddmfcc6_std', 'ddmfcc7_std', 'ddmfcc8_std', 'ddmfcc9_std', 'ddmfcc10_std', 'ddmfcc11_std', 'ddmfcc12_std', 'ddmfcc13_std', 'mfcc1_skew', 'mfcc2_skew', 'mfcc3_skew', 'mfcc4_skew', 'mfcc5_skew', 'mfcc6_skew', 'mfcc7_skew', 'mfcc8_skew', 'mfcc9_skew', 'mfcc10_skew', 'mfcc11_skew', 'mfcc12_skew', 'mfcc13_skew', 'dmfcc1_skew', 'dmfcc2_skew', 'dmfcc3_skew', 'dmfcc4_skew', 'dmfcc5_skew', 'dmfcc6_skew', 'dmfcc7_skew', 'dmfcc8_skew', 'dmfcc9_skew', 'dmfcc10_skew', 'dmfcc11_skew', 'dmfcc12_skew', 'dmfcc13_skew', 'ddmfcc1_skew', 'ddmfcc2_skew', 'ddmfcc3_skew', 'ddmfcc4_skew', 'ddmfcc5_skew', 'ddmfcc6_skew', 'ddmfcc7_skew', 'ddmfcc8_skew', 'ddmfcc9_skew', 'ddmfcc10_skew', 'ddmfcc11_skew', 'ddmfcc12_skew', 'ddmfcc13_skew', 'mfcc1_kurt', 'mfcc2_kurt', 'mfcc3_kurt', 'mfcc4_kurt', 'mfcc5_kurt', 'mfcc6_kurt', 'mfcc7_kurt', 'mfcc8_kurt', 'mfcc9_kurt', 'mfcc10_kurt', 'mfcc11_kurt', 'mfcc12_kurt', 'mfcc13_kurt', 'dmfcc1_kurt', 'dmfcc2_kurt', 'dmfcc3_kurt', 'dmfcc4_kurt', 'dmfcc5_kurt', 'dmfcc6_kurt', 'dmfcc7_kurt', 'dmfcc8_kurt', 'dmfcc9_kurt', 'dmfcc10_kurt', 'dmfcc11_kurt', 'dmfcc12_kurt', 'dmfcc13_kurt', 'ddmfcc1_kurt', 'ddmfcc2_kurt', 'ddmfcc3_kurt', 'ddmfcc4_kurt', 'ddmfcc5_kurt', 'ddmfcc6_kurt', 'ddmfcc7_kurt', 'ddmfcc8_kurt', 'ddmfcc9_kurt', 'ddmfcc10_kurt', 'ddmfcc11_kurt', 'ddmfcc12_kurt', 'ddmfcc13_kurt']
        feature_cols = acoustic_feats + mfcc_feats
    else:
        feature_cols = list(range(X.shape[1])) 

    df = pd.DataFrame(data=data, columns=feature_cols + ['y', 'subject_id', 'sample_id', 'gender'])
    if ifm_nifm.startswith('nifm'):
        import get_nifm_features
        df = get_nifm_features.aggregate_windows(df)

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
    print("Identified {} files, of which {} M and {} F.".format(df.shape[0],
                                                                             len([i for i in df['gender'] if
                                                                                  i == 1]),
                                                                             len([i for i in df['gender'] if
                                                                                  i == 0])))
    
    # if ifm_nifm.startswith('nifm'):
    #     import get_nifm_features
    #     # df = get_nifm_features.reduce_dims(df, len(df.columns)-5)
    #     df = get_nifm_features.aggregate_windows(df)

    df.to_csv(os.path.join(store_location[0], f"{store_location[1]}_{ifm_nifm}.csv"), index=False)
    print("Combined dataframe is saved to",os.path.join(store_location[0], f"{store_location[1]}_{ifm_nifm}.csv"))


def create_features(dataset, ifm_nifm):
    dir, store_location = get_dirs(dataset)

    files, HC_id_list, PD_id_list = load_files(dir)
    parent_dir = os.path.dirname(dir[:-1])
    genderinfo = pd.read_csv(os.path.join(parent_dir, 'gender.csv'), header=0)

    print("Found {} speakers, of which {} PD and {} HC.".format(len(HC_id_list) + len(PD_id_list), len(PD_id_list),
                                                                len(HC_id_list)))

    X, y, subj_id, sample_id, gender = [], [], [], [], []

    for id, file in enumerate(files):
        print("Processing file [{}/{}]".format(id+1, len(files)))
        path_to_file = os.path.join(dir, file) + '.wav'

        if ifm_nifm.startswith('ifm'):
            import get_ifm_features
            features = get_ifm_features.get_features(path_to_file)
        elif ifm_nifm.startswith('nifm'):
            import get_nifm_features
            features = get_nifm_features.get_features(path_to_file)
        elif ifm_nifm.startswith('spec'):
            import get_ifm_features
            features = get_ifm_features.get_spectrograms(path_to_file)
    
        X.extend(features)
        y.extend([1 if file[:2] == 'PD' else 0] * features.shape[0])
        subj_id.extend([file[-4:]] * features.shape[0])
        sample_id.extend([id] * features.shape[0])
        gender.extend([genderinfo.loc[genderinfo['ID']==int(file[-4:]), 'Sex'].item()] * features.shape[0])
        if id % 20 == 0 and id > 0:
            save_intermediate_results(X, y, subj_id, sample_id, gender, ifm_nifm, store_location, id)
            X, y, subj_id, sample_id, gender = [], [], [], [], []  # Start with fresh variables
        if id == len(files)-1:
            save_intermediate_results(X, y, subj_id, sample_id, gender, ifm_nifm, store_location, id)

    combine_dfs(store_location, ifm_nifm)
