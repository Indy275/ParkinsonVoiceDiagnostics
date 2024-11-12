import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoProcessor, HubertModel
import librosa
from sklearn.preprocessing import StandardScaler
import os.path
import numpy as np
import pandas as pd
import configparser

from eval import evaluate_predictions
from file_util import load_files, get_dirs
from data_util import get_samples

config = configparser.ConfigParser()
config.read('settings.ini')

recreate_features = config.getboolean('RUN_SETTINGS', 'recreate_features')

def PTM():
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
    return model


def create_PTM():
    model = PTM()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoder.layers[-2:].parameters():  # Unfreeze last layers
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, scheduler, criterion


def load_data(dataset):
    fdir, store_location = get_dirs(dataset)
    files, HC_id_list, PD_id_list = load_files(fdir)
    parent_dir = os.path.dirname(fdir[:-1])
    genderinfo = pd.read_csv(os.path.join(parent_dir, 'gender.csv'), header=0)
    return fdir, store_location, files, genderinfo, HC_id_list, PD_id_list

def get_features(fdir, store_location, files, genderinfo, audio_length=150000, sr=16000):
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

    X, y, subj_id, sample_id, gender = [], [], [], [], []
    for id, file in enumerate(files):
        if id % 10 == 0:
            print("Processing file [{}/{}]".format(id+1, len(files)))
        path_to_file = os.path.join(fdir, file) + '.wav'
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
       
    X = np.vstack(X).astype(float)
    y = np.array(y).reshape(-1, 1)
    subj_id = np.array(subj_id).reshape(-1, 1)
    sample_id = np.array(sample_id).reshape(-1, 1)
    gender = np.array(gender).reshape(-1, 1)
    n_features = X.shape[1]
    
    data = np.hstack((X, y, subj_id, sample_id, gender))
    base_df = pd.DataFrame(data=data, columns=list(range(X.shape[1])) + ['y', 'subject_id', 'sample_id', 'gender'])
    base_df.to_csv(os.path.join(store_location[0], f"{store_location[1]}_HuBERT.csv"), index=False)
    
    return base_df, n_features


def split_train_test(base_df, HC_id_list, PD_id_list, n_features):
    train, test = get_samples(1, HC_id_list, PD_id_list, int((len(HC_id_list)+len(PD_id_list))*0.3), base_df)
    base_X_train = train.loc[:, base_df.columns[:n_features]].values
    base_X_test = test.loc[:, base_df.columns[:n_features]].values
    base_y_train = train.loc[:, 'y'].values
    base_y_test = test.loc[:, 'y'].values

    base_X_train = base_X_train.astype(np.float32)
    base_y_train = base_y_train.astype(np.int32)
    base_X_test = base_X_test.astype(np.float32)
    base_y_test = base_y_test.astype(np.int32)

    scaler = StandardScaler()
    base_X_train = scaler.fit_transform(base_X_train)
    base_X_test = scaler.transform(base_X_test)

    base_X_train = torch.tensor(base_X_train).to(torch.float32)
    base_y_train = torch.tensor(np.squeeze(base_y_train)).type(torch.LongTensor)
    base_X_test = torch.tensor(base_X_test).to(torch.float32)
    base_y_test = torch.tensor(np.squeeze(base_y_test)).type(torch.LongTensor)

    return base_X_train, base_X_test, base_y_train, base_y_test, test


def train_ptm(model, optimizer, scheduler, criterion, train_loader, test_df, x_test, y_test, num_epochs=6):
    for epoch in range(num_epochs):
        model.train()   

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(batch_data)
            hidden_states = outputs.last_hidden_state
            logits = model.classifier(hidden_states.mean(dim=1)) 
            loss = criterion(logits, batch_labels) 
            # print(logits, batch_labels, loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Evaluation
        base_X_test = x_test
        base_y_test = y_test
        # base_X_test = test_df.loc[:, test_df.columns[:-4]].values
        # base_X_test = base_X_test.astype(np.float32)
        # base_X_test = torch.tensor(base_X_test).to(torch.float32)
        # base_y_test = test_df.loc[:, 'y'].values
        # base_y_test = torch.tensor(np.squeeze(base_y_test)).type(torch.LongTensor)

        model.eval()
        with torch.no_grad():
            outputs = model(base_X_test)
            hidden_states = outputs.last_hidden_state
            logits = model.classifier(hidden_states.mean(dim=1))
            val_loss = criterion(logits, base_y_test)
            logits = torch.argmax(logits, 1)
        test_df.loc[:, 'preds'] = logits.numpy()

        metr = evaluate_predictions(f'PTM', test_df.loc[:, 'y'].values, test_df)
        file_metrics, _ = zip(*metr)
        print(file_metrics)

        print(f'Epoch [{epoch+1}/{num_epochs}]: Loss: {loss.item():.4f}, Val acc: {file_metrics[0]:.4f}, Val loss: {val_loss.item():.4f}') 

        # print("Mean Acc:", round(sum(i[0] for i in file_metrics), 3))
        # print("Mean AUC:", round(sum(i[1] for i in file_metrics), 3))
        # print("Mean Sens:", round(sum(i[2] for i in file_metrics), 3))
        # print("Mean Spec:", round(sum(i[3] for i in file_metrics), 3))

        # fmetrics_df = pd.DataFrame(file_metrics, columns=['Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity'])
        # fmetrics_df.to_csv(os.path.join('experiments', f'PTM_{dataset}.csv'), index=False)

def run_ptm(dataset):
    sr = 16000
    audio_length = 60000

    fdir, store_location, files, genderinfo, HC_id_list, PD_id_list = load_data(dataset)
    
    if recreate_features:
        base_df, n_features = get_features(fdir,  store_location, files, genderinfo, audio_length, sr)
    else:
        base_df = pd.read_csv(os.path.join(store_location[0], f"{store_location[1]}_HuBERT.csv"))
        n_features = len(base_df.columns) - 4

    base_X_train, base_X_test, base_y_train, base_y_test, test = split_train_test(base_df, HC_id_list, PD_id_list, n_features)
    
    train_dataset = TensorDataset(base_X_train, base_y_train)
    train_loader = DataLoader(train_dataset, shuffle=True) 

    print("Data generated; now training the model")
    
    model, optimizer, scheduler, criterion = create_PTM()
    

    train_ptm(model, optimizer, scheduler, criterion, train_loader, test, base_X_test, base_y_test)
    
    