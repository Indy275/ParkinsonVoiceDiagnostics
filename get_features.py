import librosa
import librosa.display
import numpy as np
sr = 44100  # Sampling rate
frame_size = 1024  # Number of samples per frame
frame_step = 256  # Number of samples between successive frames
fmin = 20  # Min frequency to use in Mel spectrogram
fmax = sr // 2  # Max frequency
n_mels = 28  # Number of mel bands to generate


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
    spectrogram = librosa.feature.melspectrogram(y=x,
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


def get_features(x):
    feature_list = []
    x = x[len(x)%frame_size:]

    # MFCC training data
    mfcc = get_mfcc(x, sample_rate=sr)

    # Mel spectrogram training data
    mels_list = []
    mels = calc_mels(x)
    mels_list.append(mels)
    mels_array = np.array(mels_list)
    mels_array = mels_array.squeeze()

    feature_list.extend(np.concatenate((mfcc, mels_array), axis=1))
    return np.array(feature_list)

