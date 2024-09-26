import os.path

import librosa
import librosa.display
import numpy as np
import random
from disvoice.prosody import Prosody
from disvoice.phonation import Phonation

from data_util.file_util import load_files
import get_features

sr = 44100  # Sampling rate
frame_size = 1024  # Number of samples per frame
frame_step = 256  # Number of samples between successive frames
fmin = 20  # Min frequency to use in Mel spectrogram
fmax = sr // 2  # Max frequency
n_mels = 28  # Number of mel bands to generate

dataset = 'italian2'

def make_train_test_split(id_list, test_size=0.3, seed=1):
    """
    Divide a list into a training set and a testing set according to a given test set percentage
    """
    random.Random(seed).shuffle(id_list)
    cut = int(test_size * len(id_list))
    train_set = id_list[cut:]
    test_set = id_list[:cut]
    return train_set, test_set


def preprocess_data(dataset):
    if dataset.lower() == 'neurovoz':
        dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\NeuroVoz\\audios\\"
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
    elif dataset.lower() == 'italian2':  # file-level phonation features
        dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\ItalianPD\\records\\"
        store_location = 'C:\\Users\\INDYD\\Documents\\RAIVD_data\\preprocessed_data\\Italian2_preprocessed\\'
    if not os.path.exists(store_location):
        os.makedirs(store_location)

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
        x, _ = librosa.core.load(os.path.join(dir, file)+ '.wav', sr=16000)

        # Prosodic features
        # prosody = Prosody()
        # prosodic_features = prosody.extract_features_file(os.path.join(dir, file)+ '.wav', static=False, plots=False, fmt="npy")
        # prosodic_features = prosodic_features.reshape(1,-1)

        # Phonation features
        phon = Phonation()
        features = phon.extract_features_file(os.path.join(dir, file)+ '.wav', static=True, plots=False, fmt="npy")

        X.extend(features)

        status = file[:2]
        if status == 'PD':
            indication = 1
            prevalence.append('PD')
        else:
            indication = 0  # 'HC'
            prevalence.append('HC')
        # y.extend([indication] )
        # subj_id.extend([file[-4:]] )
        # sample_id.extend([id_count] )
        # train_data.extend([str(file[-4:]) in PD_train + HC_train] )
        y.extend([indication] * features.shape[0])
        subj_id.extend([file[-4:]] * features.shape[0])
        sample_id.extend([id_count] * features.shape[0])
        train_data.extend([str(file[-4:]) in PD_train + HC_train] * features.shape[0])
        id_count += 1

    X = np.vstack(X)
    y = np.array(y)
    subj_id = np.array(subj_id)
    sample_id = np.array(sample_id)
    train_data = np.array(train_data)

    print("Of the {} files, {} are from PD patients and {} are from HC".format(len(prevalence),
                                                                               len([i for i in prevalence if i == 'PD']),
                                                                               len([i for i in prevalence if i == 'HC'])))
    print(X.shape, y.shape, subj_id.shape, sample_id.shape, train_data.shape)

    np.save(store_location+'X.npy', X)
    np.save(store_location+'y.npy', y)
    np.save(store_location+'subj_id.npy', subj_id)
    np.save(store_location+'sample_id.npy', sample_id)
    np.save(store_location+'train_data.npy', train_data)

preprocess_data(dataset)