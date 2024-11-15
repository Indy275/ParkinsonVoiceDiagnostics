import librosa
import torch
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.decomposition import PCA
import soundfile as sf
import tempfile
from transformers import AutoProcessor, HubertModel

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

def get_features_vggish(audio_path):
    global model
    def preprocess_audio(audio_path):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav_file:
                wav_path = temp_wav_file.name  # Get the temporary file path
                y, sr = sf.read(audio_path)  # Read MP3 using soundfile
                min_audio_length = sr * 2  # at least 2 seconds of audio
                if len(y) < min_audio_length:
                    y = np.pad(y, int(np.ceil((min_audio_length - len(y))/2)))
                sf.write(wav_path, y, sr, format='wav')  # Write to temporary WAV file
        return wav_path

    wav_path = preprocess_audio(audio_path)
    embeddings = model.forward(wav_path)
    return embeddings.detach().numpy()[:10, :]


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