import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoProcessor, HubertModel
import librosa
import librosa 
import os.path
import numpy as np
import pandas as pd

from eval import evaluate_predictions
from file_util import load_files, get_dirs
from data_util import get_samples

def create_PTM():
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

    hidden_size1 = 1024
    hidden_size2 = 512
    dropout_prob=0.25
    model.classifier = nn.Sequential(
        nn.Linear(model.config.hidden_size, hidden_size1),
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Linear(hidden_size1, hidden_size2),
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Linear(hidden_size2, 2)
    )
    return model, processor

def train_ptm(dataset):
    sr = 16000
    dir, store_location = get_dirs(dataset)

    files, HC_id_list, PD_id_list = load_files(dir)
    parent_dir = os.path.dirname(dir[:-1])
    genderinfo = pd.read_csv(os.path.join(parent_dir, 'gender.csv'), header=0)
    print(dir, store_location, files)
    model, processor = create_PTM()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    X, y, subj_id, sample_id, gender = [], [], [], [], []
    audio_length = 100000
    for id, file in enumerate(files):
        print("Processing file {} of {}".format(id+1, len(files)))
        path_to_file = os.path.join(dir, file) + '.wav'
        x, _ = librosa.core.load(path_to_file, sr=sr)
        if len(x) < audio_length:
            x = np.pad(x, int(np.ceil((audio_length - len(x))/2)))
        x = x[:audio_length]
        features = processor(x, return_tensors="pt", sampling_rate=sr).input_values
        X.extend(features)
        y.extend([1 if file[:2] == 'PD' else 0] * features.shape[0])
        subj_id.extend([file[-4:]] * features.shape[0])
        sample_id.extend([id] * features.shape[0])
        gender.extend([genderinfo.loc[genderinfo['ID']==int(file[-4:]), 'Sex'].item()] * features.shape[0])
       

    X = np.vstack(X)
    y = np.array(y).reshape(-1, 1)
    subj_id = np.array(subj_id).reshape(-1, 1)
    sample_id = np.array(sample_id).reshape(-1, 1)
    gender = np.array(gender).reshape(-1, 1)
    data = np.hstack((X, y, subj_id, sample_id, gender))
    X = X.astype(float)
    base_df = pd.DataFrame(data=data, columns=list(range(X.shape[1])) + ['y', 'subject_id', 'sample_id', 'gender'])
    
    n_features = X.shape[1]
    train, test = get_samples(1, HC_id_list, PD_id_list, int((len(HC_id_list)+len(PD_id_list))*0.25), base_df)
    base_X_train = train.loc[:, base_df.columns[:n_features]].values
    base_X_test = test.loc[:, base_df.columns[:n_features]].values
    base_y_train = train.loc[:, 'y'].values
    base_y_test = test.loc[:, 'y'].values
    base_X_train = base_X_train.astype(np.float32)
    base_y_train = base_y_train.astype(np.int32)
    base_X_test = base_X_test.astype(np.float32)


    base_X_train = torch.tensor(base_X_train).to(torch.float32)
    base_y_train = torch.tensor(np.squeeze(base_y_train)).type(torch.LongTensor)
    base_X_test = torch.tensor(base_X_test).to(torch.float32)
    train_dataset = TensorDataset(base_X_train, base_y_train)
    train_loader = DataLoader(train_dataset) 

    num_epochs=10
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoder.layers[-3:].parameters():  # Unfreeze last 3 transformer layers
        param.requires_grad = True

    # Training loop (simplified for one batch, repeat in an epoch loop for full training)
    for epoch in range(num_epochs):
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(batch_data)
            hidden_states = outputs.last_hidden_state
            logits = model.classifier(hidden_states.mean(dim=1)) 
            loss = criterion(logits, batch_labels) 
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}]: Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(base_X_test)
        print("base_preds",outputs)
        hidden_states = outputs.last_hidden_state
        logits = model.classifier(hidden_states.mean(dim=1)) 
        print(logits)
        logits = torch.argmax(logits, 1)
        print(logits)

    test.loc[:, 'preds'] = logits.numpy()

    all_metrics = evaluate_predictions(f'PTM', test.loc[:, 'y'].values, test)
    