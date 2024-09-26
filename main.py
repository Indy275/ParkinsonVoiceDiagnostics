import librosa
import matplotlib.pyplot as plt
import numpy as np

path = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\NeuroVoz\\audios\\"
filename = path + "HC_A1_0034" + ".wav"

sr = 16000  # Sampling rate
frame_size = 500  # Number of samples per frame
frame_step = 250  # Number of samples between successive frames
fmin = 20  # Min frequency to use in Mel spectrogram
fmax = sr // 2  # Max frequency; Nyquist frequency
n_mels = 28  # Number of mel bands to generate

x, _ = librosa.core.load(filename, sr=sr)
plt.figure(figsize=(10, 6))

plt.subplot(311)
plt.plot(x)
plt.grid(True, which='major', axis='x')
plt.tight_layout()

plt.subplot(312)
spectrogram = librosa.feature.melspectrogram(x[:-1],
                                             sr=sr,
                                             n_mels=n_mels,
                                             hop_length=frame_step,
                                             n_fft=frame_size,
                                             fmin=fmin,
                                             fmax=fmax)
spectrogram = librosa.power_to_db(spectrogram).astype(np.float32)
librosa.display.specshow(spectrogram,
                         x_axis='time',
                         y_axis='mel',
                         sr=sr,
                         hop_length=frame_step,
                         fmax=fmax)
plt.tight_layout()
plt.show()