import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from data_util import load_data

import configparser

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'settings.ini'))
data_dir = config['DATA_SETTINGS']['data_dir']

dataset1 = 'NeuroVoztdu'  # NeuroVoz ItalianPD CzechPD test PCGITA
dataset2 = 'PCGITAtdu'
ifm_or_nifm = 'ifm'

df1, n_features = load_data(dataset1, ifm_or_nifm)
df1['ID'] = 'Italian'
df2, n_features = load_data(dataset2, ifm_or_nifm)
df2['ID'] = 'PC-GITA'

df = pd.concat([df1, df2])
X = df.iloc[:, :n_features]
y = df['y']
id = df['ID']

print(X.shape, y.shape)
X[np.isnan(X)] = 0

tsne = TSNE(n_components=2, verbose=0, perplexity=4, max_iter=400)
tsne_results = tsne.fit_transform(X)

df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two', 'y'])
df['tsne-2d-one'] = tsne_results[:, 0]
df['tsne-2d-two'] = tsne_results[:, 1]
df['y'] = y.values
df['ID'] = id.values

plt.figure(figsize=(8, 8))
ax = sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="ID", palette=sns.color_palette("bright", 2), data=df, alpha=0.75)
ax.set(xlabel='First dimension', ylabel='Second dimension', title=f'{dataset1}, {dataset2}: 176 features compressed into two dimensions')
plt.show()
