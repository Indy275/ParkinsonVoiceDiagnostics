import librosa
import matplotlib.pyplot as plt


sr = 8000  # Sampling rate
frame_size = 500  # Number of samples per frame
frame_step = 250  # Number of samples between successive frames
fmin = 20  # Min frequency to use in Mel spectrogram
fmax = sr // 2  # Max frequency; Nyquist frequency
n_mels = 28  # Number of mel bands to generate
duration = 20  # time in seconds
nr_samples = sr * duration  # Number of samples to use for plotting
nr_samples = nr_samples - nr_samples % frame_step
wlen = 2000

plt.subplot(414)
# spectrogram /= spectrogram.max()  # Normalize spectrogram
# spectrogram = np.log10(spectrogram)  # Take log
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
plt.xlim((0, duration))
plt.tight_layout()
plt.show()