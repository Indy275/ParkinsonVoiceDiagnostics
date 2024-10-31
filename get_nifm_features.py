import librosa
import torch
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.decomposition import PCA
from transformers import AutoProcessor, HubertModel

def get_features(path_to_file):
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    
    sr = 16000
    x, _ = librosa.core.load(path_to_file, sr=sr)
    input_values = processor(x, return_tensors="pt", sampling_rate=sr).input_values
    with torch.no_grad():
        hidden_states = model(input_values).last_hidden_state
    embedding = hidden_states.detach().numpy()
    return np.squeeze(embedding)


def reduce_dims(df, n_features, n_components=150):
    pca = PCA(n_components=n_components)
    pca.fit(df.iloc[:, :n_features])
    transformed_feats = pca.transform(df.iloc[:, :n_features])
    df2 = pd.DataFrame(data=transformed_feats)
    df2.columns = ['PC{}'.format(i) for i in range(1, n_components+1)]
    df2['y'] = df['y']
    df2['subject_id'] = df['subject_id']
    df2['sample_id'] = df['sample_id']
    df2['gender'] = df['gender']

    # plt.plot(list(range(n_components)), pca.explained_variance_ratio_.cumsum())
    # # plt.vlines(150, 0, 1 , colors='red', linestyles='dashed')
    # plt.xlabel('Number of components')
    # plt.ylabel('Explained variance')
    # plt.title('Explained variance by number of components')
    # plt.tight_layout()
    # plt.show()

    print(f"Explained variance with {n_components} components:", pca.explained_variance_ratio_.cumsum()[n_components-1])

    return df2

def aggregate_windows(df):
    feature_cols = df.columns[:-4]
    metadata_cols = df.columns[-4:]
    metadata_df = df.loc[:, metadata_cols]
    metadata_df = metadata_df.drop_duplicates(subset=['subject_id'])

    def std(x): return np.std(x)
    def skew(x): return ss.skew(x)
    def kurt(x): return ss.kurtosis(x)
    aggregations = ['mean', std, skew, kurt]
    aggregated_df = df.groupby('subject_id').agg({col: aggregations for col in feature_cols}).fillna(0)
    
    aggregated_df.columns = ['_'.join(str(col)).strip() for col in aggregated_df.columns.values]
    aggregated_df = pd.concat([aggregated_dfreset_index(drop=True, inplace=True), metadata_dfreset_index(drop=True, inplace=True)], ignore_index=True, axis=1)
    aggregated_df.columns = list(aggregated_df.columns[:-4]) + list(metadata_df.columns)
    return aggregated_df