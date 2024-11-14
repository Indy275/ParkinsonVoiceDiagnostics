import numpy as np
import librosa
import librosa.display as display
import scipy.stats

import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt

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
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)

    f1_mean = np.array([f1 for f1 in f1_list if str(f1) != 'nan']).mean()
    f2_mean = np.array([f2 for f2 in f2_list if str(f2) != 'nan']).mean()
    f3_mean = np.array([f3 for f3 in f3_list if str(f3) != 'nan']).mean()

    formants_mean = np.hstack((f1_mean, f2_mean, f3_mean)).reshape(1, -1)
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


def get_features(path_to_file):
    sr = 16000
    x, _ = librosa.core.load(path_to_file, sr=sr)
    sound = parselmouth.Sound(path_to_file)
    sound = sound.resample(sr)
    f0_feats = get_f0(sound)
    formants = measureFormants(sound, fmin, fmax)
    jitter_shimmer = measurePitch(sound, fmin, fmax)
    mfcc_feats = get_mfcc(x)

    # [6, 5, 6, 3, 156]
    ifm_feats = np.hstack((f0_feats, jitter_shimmer, formants, mfcc_feats))
    ifm_feats = np.nan_to_num(ifm_feats)
    return ifm_feats


def get_spectrograms(file_path):
    sr = 16000
    start_time = sr * 1
    audio_length = sr * 2 # 5 seconds of audio
    y, sr = librosa.load(file_path, sr=sr)  
    if len(y) < audio_length+start_time:
            y = np.pad(y, int(np.ceil((start_time+audio_length - len(y))/2)))
    y = y[start_time:start_time+audio_length]

    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=128, n_mels=80)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # fig, ax = plt.subplots()
    # img = display.specshow(spectrogram_db, y_axis='mel', x_axis='time', ax=ax)
    # ax.set(title='Mel spectrogram display')
    # fig.colorbar(img, ax=ax, format="%+2.f dB")
    # plt.show()
    # y, sr = librosa.load(file_path, sr=sr)  
    # fig, ax = plt.subplots()
    # M = librosa.feature.melspectrogram(y=y, sr=sr)
    # M_db = librosa.power_to_db(M, ref=np.max)
    # img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax)
    # ax.set(title='Mel spectrogram display')
    # fig.colorbar(img, ax=ax, format="%+2.f dB")
    # # plt.show()
    return spectrogram_db.T