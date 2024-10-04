import numpy as np
import librosa
import scipy.stats

from disvoice.prosody import Prosody
from disvoice.phonation import Phonation

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

    return array of float
            the normalized array
    """
    eps = 1e-5
    return (x - np.mean(x)) / (eps + np.std(x))


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

    return list of list of float
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
    mfcc = librosa.feature.mfcc(y=raw_data, sr=sample_rate, n_mfcc=13, n_fft=frame_size,
                                hop_length=frame_step)
    mfccd = librosa.feature.delta(data=mfcc, order=1)
    mfccdd = librosa.feature.delta(data=mfcc, order=2)
    mfcc_matrix = np.vstack((mfcc, mfccd, mfccdd))
    return mfcc_matrix.T


def get_mel_features(path_to_file, static):
    sr = 16000
    x, _ = librosa.core.load(path_to_file, sr=sr)
    x = x[len(x) % frame_size:]

    # MFCC training data
    mfcc = get_mfcc(x, sample_rate=sr)
    if static:
        mean_mfcc = np.mean(mfcc, axis=0)
        std_mfcc = np.std(mfcc, axis=0)
        skew_mfcc = scipy.stats.skew(mfcc, axis=0)
        kurt_mfcc = scipy.stats.kurtosis(mfcc, axis=0)
        mfcc = np.hstack((mean_mfcc, std_mfcc, skew_mfcc, kurt_mfcc)).reshape((1,-1))

    return mfcc


def get_prosodic_features(path_to_file):
    # Prosodic features
    prosody = Prosody()
    prosodic_features = prosody.extract_features_file(path_to_file, static=False, plots=False, fmt="npy")
    prosodic_features = prosodic_features.reshape(1, -1)
    return prosodic_features


def get_phonation_features(path_to_file, static_or_dynamic):
    # Phonation features
    phon = Phonation()
    phonation_features = phon.extract_features_file(path_to_file, static=static_or_dynamic, plots=False, fmt="npy")
    print("phon",static_or_dynamic, np.shape(phonation_features))
    return phonation_features


def get_features(path_to_file, static_or_dynamic):
    get_phonation_features(path_to_file, static_or_dynamic)
    return get_mel_features(path_to_file, static_or_dynamic)
    # return get_phonation_features(path_to_file, static_or_dynamic)
