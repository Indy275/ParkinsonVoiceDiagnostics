import numpy as np
import librosa
import librosa.display as display
import matplotlib.pyplot as plt

import scipy.stats

import parselmouth
from parselmouth.praat import call

sr = 16000  # Sampling rate
frame_size = int(0.04 * sr)  # Number of samples per frame
frame_step = int(0.02 * sr)  # Number of samples between successive frames
fmin = 75  # Min frequency to use in Mel spectrogram
fmax = 300  # Max frequency


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


def get_f0(sound):
    pitch = call(sound, "To Pitch", 0.0, fmin, fmax)
    meanF0 = call(pitch, "Get mean", 0, 0, 'Hertz')
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, 'Hertz')
    minF0 = call(pitch, "Get minimum", 0, 0, 'Hertz', 'parabolic')
    maxF0 = call(pitch, "Get maximum", 0, 0, 'Hertz', 'parabolic')

    pitch_values = pitch.selected_array['frequency']
    df0 = np.diff(pitch_values, 1)
    ddf0 = np.diff(df0, 1)

    meandF0 = df0.mean()
    meanddF0 = ddf0.mean()
    ['F0_mean', 'F0_std','F0_min', 'F0_std','F0_max', 'dF0_mean', 'ddF0_mean']
    measuresF0 = np.hstack((meanF0, stdevF0, minF0, maxF0, meandF0, meanddF0)).reshape((1, -1))
    return measuresF0


def measurePitch(sound, f0min, f0max):
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    jit_shim_measures = np.hstack((localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter,
                                   localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer))
    return jit_shim_measures.reshape(1, -1)


def measureFormants(sound, f0min, f0max):
    formants = sound.to_formant_burg(time_step=0.010, maximum_formant=5000)
        
    f1_list, f2_list, f3_list  = [], [], []
    f1bw_list, f2bw_list, f3bw_list = [], [], []
    for t in formants.ts():
        f1 = formants.get_value_at_time(1, t)
        f2 = formants.get_value_at_time(2, t)
        f3 = formants.get_value_at_time(3, t)
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)

        f1bw = formants.get_bandwidth_at_time(1, t)
        f2bw = formants.get_bandwidth_at_time(2, t)
        f3bw = formants.get_bandwidth_at_time(3, t)
        f1bw_list.append(f1bw)
        f2bw_list.append(f2bw)
        f3bw_list.append(f3bw)

    f1_mean = np.array([f1 for f1 in f1_list if str(f1) != 'nan']).mean()
    f2_mean = np.array([f2 for f2 in f2_list if str(f2) != 'nan']).mean()
    f3_mean = np.array([f3 for f3 in f3_list if str(f3) != 'nan']).mean()

    f1_bw_mean = np.array([f1 for f1 in f1bw_list if str(f1) != 'nan']).mean()
    f2_bw_mean = np.array([f2 for f2 in f2bw_list if str(f2) != 'nan']).mean()
    f3_bw_mean = np.array([f3 for f3 in f3bw_list if str(f3) != 'nan']).mean()

    formants_mean = np.hstack((f1_mean, f2_mean, f3_mean, f1_bw_mean, f2_bw_mean, f3_bw_mean)).reshape(1, -1)
    return formants_mean


def get_mfcc(x):
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13, n_fft=frame_size,
                                hop_length=frame_step)
    mfccd = librosa.feature.delta(data=mfcc, order=1, mode='nearest')
    mfccdd = librosa.feature.delta(data=mfcc, order=2, mode='nearest')
    mfcc_matrix = np.vstack((mfcc, mfccd, mfccdd))
    mfcc = mfcc_matrix.T
    
    mean_mfcc = np.mean(mfcc, axis=0)
    std_mfcc = np.std(mfcc, axis=0)
    skew_mfcc = scipy.stats.skew(mfcc, axis=0)
    kurt_mfcc = scipy.stats.kurtosis(mfcc, axis=0)

    mfcc = np.hstack((mean_mfcc, std_mfcc, skew_mfcc, kurt_mfcc)).reshape((1, -1))
   
    return mfcc


def get_features(x):
    sound = parselmouth.Sound(x, sampling_frequency=sr)
    f0_feats = get_f0(sound)
    jitter_shimmer = measurePitch(sound, fmin, fmax)
    formants = measureFormants(sound, fmin, fmax)
    mfcc_feats = get_mfcc(x)

    # [6, 5, 6, 6, 156]
    ifm_feats = np.hstack((f0_feats, jitter_shimmer, formants, mfcc_feats))
    ifm_feats = np.nan_to_num(ifm_feats)
    return ifm_feats


def get_spectrograms(y):
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=64, n_mels=65)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    mels_db = (mels_db - mels_db.mean()) / mels_db.std()

    # fig, ax = plt.subplots()
    # img = display.specshow(mels_db, y_axis='mel', x_axis='time', ax=ax)
    # ax.set(title='Mel spectrogram display')
    # fig.colorbar(img, ax=ax, format="%+2.f dB")
    # plt.show()

    return mels_db.T