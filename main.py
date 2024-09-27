import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

store_location = 'C:\\Users\\INDYD\\Documents\\RAIVD_data\\preprocessed_data\\NeuroVoz_preprocessed\\'

sr = 44100  # Sampling rate
frame_size = 500  # Number of samples per frame
frame_step = 250  # Number of samples between successive frames
fmin = 20  # Min frequency to use in Mel spectrogram
fmax = sr // 2  # Max frequency; Nyquist frequency
n_mels = 28  # Number of mel bands to generate

################## To be used later on in the project, to run various pipelines easily at once. Currently not used.