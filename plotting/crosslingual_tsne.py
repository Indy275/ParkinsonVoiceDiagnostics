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

if os.getenv("COLAB_RELEASE_TAG"):  # colab
    data_dir = '/content/drive/My Drive/RAIVD_data/'
elif os.name == 'posix':  # linux
    data_dir = '/home/indy/Documents/RAIVD_data/'
elif os.name == 'nt':  # windows
    data_dir = "C:\\Users\INDYD\Documents\RAIVD_data\\"

dataset1 = 'IPVS'  # NeuroVoz IPVS PCGITA CzechPD
dataset2 = 'NeuroVoz'
dataset3 = 'PCGITA'
dataset4 = 'MDVR'
datasets = [dataset1, dataset2, dataset3, dataset4]

speech_task = 'tdu'  

ifm_or_nifm = 'ifm'

markers = ['o', 's', 'D', 'v']
cpals = ['viridis',  'cubehelix', 'magma', 'Spectral']  # 'coolwarm',
colors = plt.cm.tab20b(np.linspace(0, 1, 20))
cpals = [colors[0:2], colors[4:6], colors[8:10], colors[12:14]]  # This shows difference between datasets
# cpals = ['coolwarm'] * 4  # This shows difference between PD and HC across datasets

dfs, setnames = [], []
for dataset in datasets:
    dataset += '_' + speech_task

    df1, n_features = load_data(dataset, ifm_or_nifm)
    df1['ID'] = dataset

    print(f'{dataset}: {df1.shape}')
    dfs.append(df1)
    setnames.append(dataset)

df = pd.concat(dfs)
X = df.iloc[:, :n_features].values
X[np.isnan(X)] = 0
y = df['y']
id = df['ID']

tsne = TSNE(n_components=2, verbose=0, perplexity=4, max_iter=400)
tsne_results = tsne.fit_transform(X)

df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two', 'y'])
df['tsne-2d-one'] = tsne_results[:, 0]
df['tsne-2d-two'] = tsne_results[:, 1]
df['y'] = y.values
df['ID'] = id.values

plt.figure(figsize=(6, 6))

for dataset, cpal, m in zip(setnames, cpals, markers):
    cur_df = df[df['ID']== dataset] 
    ax = sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="y", palette=list(plt.cm.tab20(np.linspace(0, 1, 20))[setnames.index(dataset)*4:setnames.index(dataset)*4+2]), marker=m, data=cur_df, alpha=0.75, legend=False)	

setnames = [s.split('_')[0] for s in setnames]
ax.set(xlabel='First dimension', ylabel='Second dimension', title=f"tSNE visualisation of {' '.join(setnames)} \n {X.shape[1]} features compressed into two dimensions")
# labels = list(np.array([[s+' HC', s+' PD'] for s in setnames]).flatten())  # This would be great, but unfortunately seaborn does not support it
plt.legend(title='Data sample', labels=[s for s in setnames], loc='upper right')

path = r"C:\Users\INDYD\Dropbox\Uni MSc AI\Master_thesis_RAIVD\imgs"
if os.path.exists(path): # Only when running on laptop
    plt.savefig(os.path.join(path,f"tSNE_{ifm_or_nifm}_{'_'.join(setnames)}.pdf"))
    print("tSNE plot saved to "+f"{path} tSNE_{ifm_or_nifm}_{'_'.join(setnames)}.pdf")
plt.show()
