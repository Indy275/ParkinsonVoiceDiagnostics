import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)
from data_util import load_data

import configparser

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'settings.ini'))
data_dir = config['DATA_SETTINGS']['data_dir']

dataset = 'NeuroVoz'  # NeuroVoz ItalianPD CzechPD test
ifm_or_nifm = 'nifm'
file_or_window = 'file'

if ifm_or_nifm == 'ifm':
    ifm_or_nifm += '_{}'.format(file_or_window)

df, n_features = load_data(dataset, ifm_or_nifm)
df.rename(columns={'subject_id':'ID'}, inplace=True)

if dataset == 'NeuroVoz':
    metadata = pd.read_csv('C://Users/INDYD/Documents/RAIVD_data/NeuroVoz/metadata/data_pd.csv')
    updrs = metadata.loc[:, ['ID','UPDRS scale']]
    df  = df.merge(updrs, on='ID', how='left')
    updrs_scale = df['UPDRS scale']

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
df['y'] = y

if dataset == 'NeuroVoz':
    df['UPDRS scale'] = updrs_scale
    nc = len(np.unique(updrs))
    df2 = df[df['y']== 1]
    df = df[df['y'] == 0]

plt.figure(figsize=(8, 8))
ax = sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="y", palette=sns.color_palette("bright", 2), data=df, alpha=0.75)
if dataset == 'NeuroVoz':
    sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="UPDRS scale", palette=sns.color_palette("flare", n_colors=nc), data=df2, alpha=0.75, ax=ax)
ax.set(xlabel='First dimension', ylabel='Second dimension', title=f'{dataset}: 176 features compressed into two dimensions')
plt.show()
