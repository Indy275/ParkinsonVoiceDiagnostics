import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from transformers import AutoProcessor, HubertModel
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os.path
import numpy as np
import pandas as pd
import configparser

from eval import evaluate_predictions
from file_util import load_files, get_dirs
from data_util import get_samples
import soundfile as sf
import tempfile

config = configparser.ConfigParser()
config.read('settings.ini')

recreate_features = config.getboolean('RUN_SETTINGS', 'recreate_features')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')
k_folds = config.getint('EXPERIMENT_SETTINGS', 'kfolds')


class DNNModel(nn.Module):
    def __init__(self, input_size, dropout_prob=0.25):
        super(DNNModel, self).__init__()

        hidden_size1 = 1024
        hidden_size2 = 1024
        hidden_size3 = 512
        hidden_size4 = 256

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.dropout4 = nn.Dropout(dropout_prob)

        self.fc5 = nn.Linear(hidden_size4, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = torch.relu(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        x = torch.sigmoid(x)

        return x
    

def create_pt_model():
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model.classifier = DNNModel(input_size=128)
    optimizer = optim.AdamW(model.parameters(), lr=0.00005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, scheduler, criterion


def train_model(model, optimizer, scheduler, criterion, train_loader, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        acc = []
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()  # Reset gradients
            x = np.squeeze(batch_data)
            outputs = model(x)  # Get model predictions
            logits = model.classifier(outputs)
            # print(logits, batch_labels)
            _, pred = torch.max(logits.data, 1)
            
            loss = criterion(logits, batch_labels.repeat(logits.shape[0]))  # Compute the loss
            acc.append(accuracy_score(pred.numpy(), batch_labels.repeat(pred.shape[0]).numpy()))
            # print(outputs, batch_labels, loss)
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model weights
            scheduler.step()

        if print_intermediate: # and epoch % (num_epochs-1) == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]: loss: {loss.item():.4f}, lr:', round(optimizer.param_groups[0]['lr'],5),
                  ', train acc:', round(np.mean(acc), 3))


def load_data(dataset):
    fdir, store_location = get_dirs(dataset)
    files, HC_id_list, PD_id_list = load_files(fdir)
    parent_dir = os.path.dirname(fdir[:-1])
    genderinfo = pd.read_csv(os.path.join(parent_dir, 'gender.csv'), header=0)
    return fdir, store_location, files, genderinfo, HC_id_list, PD_id_list


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


def get_features(fdir, store_location, files, genderinfo, audio_length=150000, sr=16000):
    X, y, subj_id, sample_id, gender = [], [], [], [], []
    for id, file in enumerate(files):
        if id % 10 == 0:
            print("Processing file [{}/{}]".format(id+1, len(files)))
        path_to_file = os.path.join(fdir, file) + '.wav'
        x, _ = librosa.core.load(path_to_file, sr=sr)
        if len(x) < audio_length:
            x = np.pad(x, int(np.ceil((audio_length - len(x))/2)))
        x = x[:audio_length]
        features = get_features_vggish(path_to_file)
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
    base_df = pd.DataFrame(data=data, columns=list(range(X.shape[1])) + ['y', 'subject_id', 'sample_id', 'gender', 'dataset'])
    base_df.to_csv(os.path.join(store_location[0], f"{store_location[1]}_vgg.csv"), index=False)
    
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
            print(batch_data.shape, outputs.shape, batch_labels)
            hidden_states = outputs.last_hidden_state
            logits = model.classifier(hidden_states.mean(dim=1)) 
            print(hidden_states.shape, logits.shape)
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
    fdir, store_location, files, genderinfo, HC_id_list, PD_id_list = load_data(dataset)
    
    X, labels = [], []
    for i, file in enumerate(files):
        y, sr = sf.read(os.path.join(fdir, file) + '.wav', dtype='int16')  # Read MP3 using soundfile
        y = y / 32768.0  # Convert to [-1.0, +1.0]
        
        audio_length = sr * 5  # 5 seconds of audio
        if len(y) < audio_length:
            y = np.pad(y, int(np.ceil((audio_length - len(y))/2)))
        y = y[:audio_length]
        X.append(y)
        labels.extend([1 if file[:2] == 'PD' else 0])
    X = torch.tensor(np.array(X)).to(torch.float32)
    labels = torch.tensor(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train,y_train), shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test,y_test), shuffle=True)
    print(X.shape, labels.shape)
    print("Data generated; now training the model")
    model, optimizer, scheduler, criterion = create_pt_model()
    train_model(model, optimizer, scheduler, criterion, train_loader, num_epochs=2)

    print("Model trained; now evaluating")
    model.eval()
    test_acc, test_loss = [], []
    for batch_data, batch_labels in test_loader:
        x = np.squeeze(batch_data.numpy())
        outputs = model(x)  # Get model predictions
        logits = model.classifier(outputs)
        _, pred = torch.max(logits.data, 1)
        
        test_loss.append(criterion(logits, batch_labels.repeat(logits.shape[0])).detach().numpy()) # Compute the loss
        test_acc.append(accuracy_score(pred.numpy(), batch_labels.repeat(pred.shape[0]).numpy()))
    print("test acc:", np.mean(test_acc))
    print("test loss:", np.mean(test_loss))



    
    