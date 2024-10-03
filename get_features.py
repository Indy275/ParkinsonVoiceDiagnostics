import os.path
import numpy as np
import pandas as pd

from data_util.file_util import load_files
from data_util.data_util import make_train_test_split
import get_ifm_features, get_nifm_features

sr = 44100  # Sampling rate
frame_size = 1024  # Number of samples per frame
frame_step = 256  # Number of samples between successive frames
fmin = 20  # Min frequency to use in Mel spectrogram
fmax = sr // 2  # Max frequency
n_mels = 28  # Number of mel bands to generate


def get_dirs(dataset):
    if dataset.lower() == 'neurovoz':  # sample-level phonation features
        dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\NeuroVoz\\audios_A\\"
        store_location = 'C:\\Users\\INDYD\\Documents\\RAIVD_data\\preprocessed_data\\NeuroVoz_preprocessed\\'
    elif dataset.lower() == 'czech':
        dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\CzechPD\\modified_records\\"
        store_location = 'C:\\Users\\INDYD\\Documents\\RAIVD_data\\preprocessed_data\\Czech_preprocessed\\'
    elif dataset.lower() == 'test':
        dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\NeuroVoz\\subsample\\"
        store_location = 'C:\\Users\\INDYD\\Documents\\RAIVD_data\\preprocessed_data\\test_preprocessed\\'
    elif dataset.lower() == 'italian':  # sample-level phonation features
        dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\ItalianPD\\records\\"
        store_location = 'C:\\Users\\INDYD\\Documents\\RAIVD_data\\preprocessed_data\\Italian_preprocessed\\'
    else:
        print(" '{}' is not a valid data set ".format(dataset))
        return
    if not os.path.exists(store_location):
        os.makedirs(store_location)
    return dir, store_location


def create_features(dataset, ifm_nifm):
    dir, store_location = get_dirs(dataset)

    files, HC_id_list, PD_id_list = load_files(dir)

    HC_train, HC_test = make_train_test_split(HC_id_list)
    PD_train, PD_test = make_train_test_split(PD_id_list)

    print("Found {} speakers, of which {} PD and {} HC.".format(len(HC_id_list)+len(PD_id_list), len(PD_id_list), len(HC_id_list)))

    print("The train set consists of {} PD and {} HC speakers.".format(len(PD_train), len(HC_train)))
    print("The test set consists of {} PD and {} HC speakers.".format(len(PD_test), len(HC_test)))

    prevalence, train_data = [], []
    X, y, subj_id, sample_id = [], [], [], []
    id_count = 0

    for file in files:
        path_to_file =os.path.join(dir, file) + '.wav'

        if ifm_nifm[:3] == 'ifm':
            if ifm_nifm[-4:] == 'file':
                static = True
            elif ifm_nifm[-6:] == 'window':
                static = False
            else:
                print("Something went wrong:", ifm_nifm)
                return

            features = get_ifm_features.get_features(path_to_file, static)
        elif ifm_nifm[:4] == 'ifm':
            features = get_nifm_features.get_features(path_to_file)
            features = np.squeeze(features)
        else:
            print("Something went wrong:", ifm_nifm)
            return

        status = file[:2]
        if status == 'PD':
            indication = 1
            prevalence.append('PD')
        else:
            indication = 0  # 'HC'
            prevalence.append('HC')
        X.extend(features)
        y.extend([indication] * features.shape[0])
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
    df = pd.DataFrame(data=data, columns=list(range(X.shape[1]))+['y', 'subject', 'sample', 'traintest'])

    print("Of the {} files, {} are from PD patients and {} are from HC".format(len(prevalence),
                                                                               len([i for i in prevalence if i == 'PD']),
                                                                               len([i for i in prevalence if i == 'HC'])))
    print(X.shape, y.shape, subj_id.shape, sample_id.shape, train_data.shape)

    np.save(store_location+'X_{}.npy'.format(ifm_nifm), X)
    np.save(store_location+'y_{}.npy'.format(ifm_nifm), y)
    np.save(store_location+'subj_id_{}.npy'.format(ifm_nifm), subj_id)
    np.save(store_location+'sample_id_{}.npy'.format(ifm_nifm), sample_id)
    np.save(store_location+'train_data_{}.npy'.format(ifm_nifm), train_data)

    df.to_csv(store_location+"data_{}.csv".format(ifm_nifm))
