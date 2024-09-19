import os
import re
import librosa
import librosa.display
import numpy as np
import random

dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\NeuroVoz\\audios\\"

sr = 44100  # Sampling rate
frame_size = 1024  # Number of samples per frame
frame_step = 256  # Number of samples between successive frames
fmin = 20  # Min frequency to use in Mel spectrogram
fmax = sr // 2  # Max frequency; Nyquist frequency
n_mels = 28  # Number of mel bands to generate


def make_train_test_split(id_list, test_size=0.3, seed=1):
    """
    Divide a list into a training set and a testing set according to a given test set percentage
    """
    random.Random(seed).shuffle(id_list)
    cut = int(test_size * len(id_list))
    train_set = id_list[cut:]
    test_set = id_list[:cut]
    return train_set, test_set


def normalize(x):
    """
    Z-Score normalization of an array

    (subtract mean and divide by standard deviation)

    :param x : array of float
            the array to be normalized

    :return array of float
            the normalized array
    """
    eps = 0.001
    if np.std(x) != 0:
        x = (x - np.mean(x)) / np.std(x)
    else:
        x = (x - np.mean(x)) / eps
    return x


def calc_mels(x):
    """
    Calculates and returns the Mel features from a given waveform

    :param x : list
            the waveform of the audio signal of the file
    :param sr : int
            the sample rate of the media files
    :param n_mels : int
            the number of mel bands to generate
    :param frame_step : int
            the number of samples between successive frames
    :param frame_size : int
            the number of samples per frame
    :param fmin : int
             min frequency to use in Mel spectrogram
    :param fmax : int
            max frequency to use in Mel spectrogram

    :return list of list of float
            the computed Mel coefficients representing the audio file
    """
    x = np.array(x)
    spectrogram = librosa.feature.melspectrogram(y=x[:-1],
                                                 sr=sr,
                                                 n_mels=n_mels,
                                                 hop_length=frame_step,
                                                 n_fft=frame_size,
                                                 fmin=fmin,
                                                 fmax=fmax)
    mels = librosa.power_to_db(spectrogram).astype(np.float32)
    mels = normalize(mels)
    mels = mels.transpose()
    mels[np.isnan(mels)] = 0
    return mels


def get_mfcc(raw_data, sample_rate):
    mfcc_features_matrix = librosa.feature.mfcc(y=raw_data, sr=sample_rate, n_mfcc=13, n_fft=frame_size,
                                                hop_length=frame_step)
    return mfcc_features_matrix.T


files = []
for file in os.listdir(dir):
    if re.match(r"^[A-Z]{2}_[A]\d_\d+$", file[2:-4]):
        files.append(file[2:-4])

HC_id_list = [f[-4:] for f in files if f[:2] == 'HC']
PD_id_list = [f[-4:] for f in files if f[:2] == 'PD']
HC_train, HC_test = make_train_test_split(HC_id_list)
PD_train, PD_test = make_train_test_split(PD_id_list)

prevalence, train_data = [], []
mels_list, y, id = [], [], []
X_emb, y_emb = [], []
for file in files:
    x, _ = librosa.core.load(dir + file + '.wav', sr=None)

    # MFCC training data
    mfcc = get_mfcc(x, sample_rate=sr)
    # mels = mfcc

    # Mel spectrogram training data
    mels = calc_mels(x)
    mels_list.append(mels)
    status = file[:2]
    if status == 'PD':
        indication = 1
        prevalence.append('PD')
    else:
        indication = 0  # 'HC'
        prevalence.append('HC')
    y.extend([indication] * mels.shape[0])
    id.extend([file[-4:]] * mels.shape[0])
    train_data.extend([str(file[-4:]) in PD_train + HC_train] * mels.shape[0])


X = np.vstack(mels_list)
y = np.array(y)
id = np.array(id)
train_data = np.array(train_data)

print("Of the {} files, {} are from PD patients and {} are from HC".format(len(prevalence),
                                                                           len([i for i in prevalence if i == 'PD']),
                                                                           len([i for i in prevalence if i == 'HC'])))
print(X.shape, y.shape, id.shape, train_data.shape)

np.save('C:\\Users\\INDYD\\Documents\\ParkinsonVoiceDiagnostics\\NeuroVoz_preprocessed\\X.npy', X)
np.save('C:\\Users\\INDYD\\Documents\\ParkinsonVoiceDiagnostics\\NeuroVoz_preprocessed\\y.npy', y)
np.save('C:\\Users\\INDYD\\Documents\\ParkinsonVoiceDiagnostics\\NeuroVoz_preprocessed\\id.npy', id)
np.save('C:\\Users\\INDYD\\Documents\\ParkinsonVoiceDiagnostics\\NeuroVoz_preprocessed\\train_data.npy', train_data)
