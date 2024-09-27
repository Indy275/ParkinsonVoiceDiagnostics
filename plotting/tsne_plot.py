import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

store_location = 'C:\\Users\\INDYD\\Documents\\RAIVD_data\\preprocessed_data\\NeuroVoz_preprocessed\\'

X = np.load(store_location + 'X.npy')
y = np.load(store_location + 'y.npy')
subj_id = np.load(store_location + 'subj_id.npy')
sample_id = np.load(store_location + 'sample_id.npy')
train_data = np.load(store_location + 'train_data.npy')

print(y, subj_id, train_data)
print(X.shape, y.shape)

tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=400)
tsne_results = tsne.fit_transform(X)

df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two', 'y'])
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
df['y'] = y

plt.figure(figsize=(8, 8))
sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="y", palette=sns.color_palette("bright", 2), data=df)
plt.show()