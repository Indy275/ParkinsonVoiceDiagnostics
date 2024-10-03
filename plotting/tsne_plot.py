import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import configparser
import os

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'settings.ini'))
data_dir = config['DATA_SETTINGS']['data_dir']

dataset = 'Czech'  # NeuroVoz Italian Czech test
ifm_or_nifm = 'ifm'
file_or_window = 'file'

store_location = data_dir + 'preprocessed_data\\{}_preprocessed\\'.format(dataset)

X = np.load(store_location + 'X_{}_{}.npy'.format(ifm_or_nifm, file_or_window))
y = np.load(store_location + 'y_{}_{}.npy'.format(ifm_or_nifm, file_or_window))
subj_id = np.load(store_location + 'subj_id_{}_{}.npy'.format(ifm_or_nifm, file_or_window))
sample_id = np.load(store_location + 'sample_id_{}_{}.npy'.format(ifm_or_nifm, file_or_window))
train_data = np.load(store_location + 'train_data_{}_{}.npy'.format(ifm_or_nifm, file_or_window))

print(X.shape, y.shape)

tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=400)
tsne_results = tsne.fit_transform(X)

df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two', 'y'])
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
df['y'] = y

plt.figure(figsize=(8, 8))
sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="y", palette=sns.color_palette("bright", 2), data=df, alpha=0.5)
plt.show()