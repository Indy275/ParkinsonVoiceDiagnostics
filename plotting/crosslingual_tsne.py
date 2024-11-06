import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from data_util import load_data

import configparser

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'settings.ini'))
data_dir = config['DATA_SETTINGS']['data_dir']

dataset1 = 'PCGITA'  # NeuroVoz ItalianPD PCGITA
dataset2 = 'NeuroVoz'
speech_task = 'ddk'

ifm_or_nifm = 'ifm'

if speech_task == 'tdu':
    dataset1 += 'tdu'
    dataset2 += 'tdu'
elif speech_task == 'ddk':
    dataset1 += 'ddk'
    dataset2 += 'ddk'

df1, n_features = load_data(dataset1, ifm_or_nifm)
df1['ID'] = dataset1
df2, n_features = load_data(dataset2, ifm_or_nifm)
df2['ID'] = dataset2

print(f'{dataset1}: {df1.shape}\n{dataset2}: {df2.shape}')
print(df1.columns, df2.columns)

df = pd.concat([df1, df2])
X = df.iloc[:, :n_features].values
y = df['y']
id = df['ID']

print(X.shape, y.shape)
X[np.isnan(X)] = 0

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

tsne = TSNE(n_components=2, verbose=0, perplexity=4, max_iter=400)
tsne_results = tsne.fit_transform(X)

df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two', 'y'])
df['tsne-2d-one'] = tsne_results[:, 0]
df['tsne-2d-two'] = tsne_results[:, 1]
df['y'] = y.values
df['ID'] = id.values
df2 = df[df['ID']== dataset2] 
df = df[df['ID']== dataset1] 

plt.figure(figsize=(8, 8))
ax = sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="y", palette=sns.color_palette("Blues", 2), data=df, alpha=0.75)
ax = sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="y", palette=sns.color_palette("YlOrBr", 2), data=df2, alpha=0.75)
ax.set(xlabel='First dimension', ylabel='Second dimension', title=f'{dataset1}, {dataset2}: {X.shape[1]} features compressed into two dimensions')

path = r"C:\Users\INDYD\Dropbox\Uni MSc AI\Master_thesis_RAIVD\imgs"
if os.path.exists(path): # Only when running on laptop
    plt.savefig(os.path.join(path,f'tSNE_{ifm_or_nifm}_{dataset1}_{dataset2}_{speech_task}.png'))
    print("tSNE plot saved to "+f'{path}\_tSNE_{ifm_or_nifm}_{dataset1}_{dataset2}_{speech_task}.png')
plt.show()
