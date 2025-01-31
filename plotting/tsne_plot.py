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
if os.getenv("COLAB_RELEASE_TAG"):  # colab
    experiment_folder = '/content/drive/My Drive/RAIVD_data/experiments/'
else:
    cwd = os.path.abspath(os.getcwd())
    experiment_folder = os.path.join(cwd,'experiments')

from data_util import load_data

if os.getenv("COLAB_RELEASE_TAG"):  # colab
    data_dir = '/content/drive/My Drive/RAIVD_data/'
elif os.name == 'posix':  # linux
    data_dir = '/home/indy/Documents/RAIVD_data/'
elif os.name == 'nt':  # windows
    data_dir = "C:\\Users\INDYD\Documents\RAIVD_data\\"


for dataset in ['NeuroVoz', 'IPVS', 'PCGITA', 'MDVR']:
    dataset += '_tdu'
    for ifm_or_nifm in ['ifm', 'hubert0']:
        df, n_features = load_data(dataset, ifm_or_nifm)
        df.rename(columns={'subject_id':'ID'}, inplace=True)

        X = df.iloc[:, :n_features]
        y = df['y']
        id = df['ID']

        print(X.shape, y.shape)
        X[np.isnan(X)] = 0

        tsne = TSNE(n_components=2, verbose=0, perplexity=4, max_iter=400)
        tsne_results = tsne.fit_transform(X)

        df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two', 'Sample'])
        df['tsne-2d-one'] = tsne_results[:, 0]
        df['tsne-2d-two'] = tsne_results[:, 1]
        df['Sample'] = y
        df['Sample'] = df['Sample'].map({0: 'HC', 1: 'PD'})

        plt.figure(figsize=(6, 6))
        ax = sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="Sample", palette=sns.color_palette("Set2", 2), data=df, alpha=0.75)
        # ax = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y, cmap='ocean', alpha=0.75)
        ax.set(xlabel='First dimension', ylabel='Second dimension', title=f"tSNE visualisation of {' '.join([s for s in dataset.split('_')])} \n {X.shape[1]} features compressed into two dimensions")
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_folder, f'{dataset}_{ifm_or_nifm}_tsne.pdf'))
        # plt.show()
