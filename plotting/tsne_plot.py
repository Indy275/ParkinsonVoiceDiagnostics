import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data_util.data_util import load_data

import configparser
import os

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'settings.ini'))
data_dir = config['DATA_SETTINGS']['data_dir']

dataset = 'ItalianPD'  # NeuroVoz ItalianPD CzechPD test
ifm_or_nifm = 'ifm'
file_or_window = 'file'

if ifm_or_nifm == 'ifm':
    ifm_or_nifm += '_{}'.format(file_or_window)

df, n_features = load_data(dataset, ifm_or_nifm)
X = df.iloc[:, :n_features]
y = df['y']

print(X.shape, y.shape)

tsne = TSNE(n_components=2, verbose=0, perplexity=10, max_iter=400)
tsne_results = tsne.fit_transform(X)

df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two', 'y'])
df['tsne-2d-one'] = tsne_results[:, 0]
df['tsne-2d-two'] = tsne_results[:, 1]
df['y'] = y

plt.figure(figsize=(8, 8))
sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="y", palette=sns.color_palette("bright", 2), data=df, alpha=0.5)
plt.show()
