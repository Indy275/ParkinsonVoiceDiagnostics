import os
import re
import librosa
import librosa.display
import numpy as np
import random
from transformers import AutoProcessor, AutoModelForAudioClassification


dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\NeuroVoz\\audios\\"

sr = 44100  # Sampling rate
frame_size = 1024  # Number of samples per frame
frame_step = 256  # Number of samples between successive frames
fmin = 20  # Min f0 to use in Mel spectrogram
fmax = 400  # Max f0
n_mels = 28  # Number of mel bands to generate


def make_train_test_split(id_list, test_size=0.3, seed=1):
    """
    Divide a list into a training set and a testing set according to a given test set percentage
    """
    random.Random(seed).shuffle(id_list)
    cut = int(test_size * len(id_list))
    train_set = id_list[cut:]
    test_set = id_list[:cut]
    return train_set, test_set

files = []
for file in os.listdir(dir):
    if re.match(r"^[A-Z]{2}_[A]\d_\d+$", file[2:-4]):
        files.append(file[2:-4])

HC_id_list = [f[-4:] for f in files if f[:2] == 'HC']
PD_id_list = [f[-4:] for f in files if f[:2] == 'PD']
HC_train, HC_test = make_train_test_split(HC_id_list)
PD_train, PD_test = make_train_test_split(PD_id_list)

# model = TFAutoModel.from_pretrained("vumichien/nonsemantic-speech-trillsson3")


processor = AutoProcessor.from_pretrained("superb/wav2vec2-base-superb-sid")
model = AutoModelForAudioClassification.from_pretrained("superb/wav2vec2-base-superb-sid")

prevalence, train_data = [], []
mels_list, y, id = [], [], []
X_emb, y_emb = [], []
for file in files:
    x, _ = librosa.core.load(dir + file + '.wav', sr=16000)
    embeddings = model(x)

# OpenL3 embedding training data
# embeddings = openl3.get_audio_embedding(x, sr=sr, embedding_size=512)
# print(embeddings.shape)
# X_emb.append(embeddings)
# y_emb.extend([indication] * embeddings.shape[0])


X_embed = np.vstack(X_emb)
y_emb = np.array(y_emb)

print("embedding",X_embed.shape, y_emb.shape)