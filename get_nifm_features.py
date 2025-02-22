import configparser
import torch
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.decomposition import PCA
import soundfile as sf
import tempfile


config = configparser.ConfigParser()
config.read('settings.ini')
ifm_or_nifm = config['EXPERIMENT_SETTINGS']['ifm_or_nifm']
if ifm_or_nifm.startswith('hubert'):
    from transformers import AutoProcessor, HubertModel
elif ifm_or_nifm == 'vgg':
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model.eval()

sr = 16000  # Sampling rate

def get_features_vggish(y):
    global model
    def preprocess_audio(y):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav_file:
                wav_path = temp_wav_file.name  # Get the temporary file path
                sf.write(wav_path, y, sr, format='wav')  # Write to temporary WAV file
        return wav_path
    wav_path = preprocess_audio(y)
    embeddings = model.forward(wav_path).detach().numpy()
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings[:10, :]



def get_features_hubert(x, layer):
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    
    input_values = processor(x, return_tensors="pt", sampling_rate=sr).input_values
    with torch.no_grad():
        if layer == '1':
            hidden_states = model(input_values).last_hidden_state
        if layer == '0':
            hidden_states = model(input_values, output_hidden_states=True).hidden_states[0]
    embedding = hidden_states.detach().numpy()
    aggregated_emb = agg_windows(np.squeeze(embedding))
    return aggregated_emb.reshape(1, -1)


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
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    def std(x): return np.std(x)
    def skew(x): return ss.skew(x)
    def kurt(x): return ss.kurtosis(x)
    aggregations = ['mean', std, skew, kurt]
    aggregated_df = df.groupby('subject_id').agg({col: aggregations for col in feature_cols}).fillna(0)
    aggregated_df.reset_index(drop=True, inplace=True)
    metadata_df.reset_index(drop=True, inplace=True)
    aggregated_df.columns = ['_'.join(str(col)).strip() for col in aggregated_df.columns.values]
    aggregated_df = pd.concat([aggregated_df, metadata_df], ignore_index=True, axis=1)
    aggregated_df.columns = list(aggregated_df.columns[:-4]) + list(metadata_df.columns)
    return aggregated_df

def agg_windows(windows):
    """ Transforms a Nx1024 np array to a 4096-D vector with functionals for each node."""
    def mean(x): return np.mean(x, axis=0)
    def std(x): return np.std(x, axis=0)
    def skew(x): return ss.skew(x, axis=0)
    def kurt(x): return ss.kurtosis(x, axis=0)
    aggregations = [mean, std, skew, kurt]
    aggregated_windows = np.hstack([func(windows) for func in aggregations]).flatten()
    return aggregated_windows