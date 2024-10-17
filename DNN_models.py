import random
import numpy as np
import configparser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim

from eval import evaluate_predictions

config = configparser.ConfigParser()
config.read('settings.ini')
plot_fimp = config.getboolean('OUTPUT_SETTINGS', 'plot_fimp')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')


class DNNModel(nn.Module):
    def __init__(self, input_size, dropout_prob=0.2):
        super(DNNModel, self).__init__()

        hidden_size1 = 1024
        hidden_size2 = 512

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(hidden_size2, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


def train_model(model, optimizer, criterion, X_train, y_train, num_epochs=10):
    batch_size = 8
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(batch_data)  # Get model predictions
            loss = criterion(outputs, batch_labels)  # Compute the loss
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model weights

        if print_intermediate:
            print(f'Epoch [{epoch+1}/{num_epochs}]: Loss: {loss.item():.4f}')

def create_dnn_model(n_features):
    input_size = n_features
    model = DNNModel(input_size)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def run_dnn_model(X_train, X_test, y_train, y_test, df, test_indices):
    n_features = X_train.shape[1]

    X_train = torch.tensor(X_train.values).to(torch.float32)
    X_test = torch.tensor(X_test.values).to(torch.float32)
    y_train = torch.tensor(y_train.values)
    y_test = y_test.values
    model, optimizer, criterion = create_dnn_model(n_features)

    train_model(model, optimizer, criterion, X_train, y_train)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        _, predicted = torch.max(predictions.data, 1)
    df.loc[test_indices, 'preds'] = predicted.numpy()

    file_scores = evaluate_predictions('DeepNN' + 'Window', y_test, df.loc[test_indices, 'preds'].tolist())

    samples_preds = df.loc[test_indices, ['subject_id', 'preds']].groupby('subject_id').agg({'preds': lambda x: x.mode()[0]}).reset_index()
    samples_ytest = df.loc[test_indices, ['subject_id', 'y']].groupby('subject_id').agg({'y': lambda x: x.mode()[0]}).reset_index()

    subj_scores = evaluate_predictions('DeepNN'+'SubjID', samples_ytest['y'].tolist(), samples_preds['preds'].tolist())
    
    return file_scores, subj_scores


def train_tl_model(model, optimizer, criterion, base_loader, target_loader, num_epochs=3):
    target_iter = iter(target_loader) if target_loader else None

    for epoch in range(num_epochs):
        model.train()
        tgt_idx = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(base_loader):
            if target_loader:
                try:
                    tgt_sample_X[tgt_idx]
                except:
                    try:
                        tgt_sample_X, tgt_sample_y = next(target_iter)
                        tgt_sample_X[tgt_idx]
                    except:
                        tgt_idx = 0
                        target_iter = iter(target_loader)
                        tgt_sample_X, tgt_sample_y = next(target_iter)
                
                batch_X[-1] = tgt_sample_X[tgt_idx] 
                batch_y[-1] = tgt_sample_y[tgt_idx] 
                tgt_idx += 1

            optimizer.zero_grad()  # Reset gradients
            outputs = model(batch_X)  # Get model predictions
            loss = criterion(outputs, batch_y)  # Compute the loss
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model weights
        if print_intermediate:
            print(f'Epoch [{epoch+1}/{num_epochs}]: Loss: {loss.item():.4f}')


def run_dnn_tl_model(base_X_train, base_X_test, base_y_train, base_y_test, tgt_df):
    n_features = base_X_train.shape[1]

    base_X_train = torch.tensor(base_X_train.values).to(torch.float32)
    base_X_test = torch.tensor(base_X_test.values).to(torch.float32)
    base_y_train = torch.tensor(base_y_train.values)
    base_y_test = base_y_test.values
    base_model, optimizer, criterion = create_dnn_model(n_features)

    base_dataset = TensorDataset(base_X_train, base_y_train)
    base_loader = DataLoader(base_dataset, batch_size=5, shuffle=True)

    train_tl_model(base_model, optimizer, criterion, base_loader, target_loader=None)

    pos_subjs = list(tgt_df[tgt_df['y'] == 1]['subject_id'].unique())
    neg_subjs = list(tgt_df[tgt_df['y'] == 0]['subject_id'].unique())
    max_shot = min(len(pos_subjs), len(neg_subjs)) - 5

    metrics_list, metrics_grouped, n_tgt_train_samples, base_metrics = [], [], [], []
    seed = int(random.random()*10000)
    for n_shots in range(max_shot):
        model = type(base_model)(n_features)
        model.load_state_dict(base_model.state_dict())

        random.seed(seed)
        pos_train_samples = random.sample(pos_subjs, n_shots)
        random.seed(seed)
        neg_train_samples = random.sample(neg_subjs, n_shots)

        tgt_train_df = tgt_df[tgt_df['subject_id'].isin(pos_train_samples + neg_train_samples)]
        tgt_test_df = tgt_df[~tgt_df['subject_id'].isin(pos_train_samples + neg_train_samples)]

        if n_shots > 0:
            # Train model with these additional samples
            tgt_X_train = tgt_train_df.iloc[:, :n_features]
            tgt_y_train = tgt_train_df['y']

            tgt_X_train = torch.tensor(tgt_X_train.values).to(torch.float32)
            tgt_y_train = torch.tensor(tgt_y_train.values)

            tgt_dataset = TensorDataset(tgt_X_train, tgt_y_train)
            tgt_loader = DataLoader(tgt_dataset, batch_size=5, shuffle=True)

            train_tl_model(model, optimizer, criterion, base_loader, target_loader=tgt_loader)

        tgt_X_test = tgt_test_df.iloc[:, :n_features].values
        tgt_X_test = torch.tensor(tgt_X_test).to(torch.float32)
        tgt_y_test = tgt_test_df['y']

        model.eval()
        with torch.no_grad():
            base_preds = model(base_X_test)
            tgt_preds = model(tgt_X_test)
            _, base_predicted = torch.max(base_preds.data, 1)
            _, tgt_predicted = torch.max(tgt_preds.data, 1)

        tgt_test_df.loc[:, 'preds'] = tgt_predicted.numpy()

        n_tgt_train_samples.append(n_shots)
        metrics_list.append(evaluate_predictions('DNN' + '0{}shot'.format(n_shots), tgt_y_test, tgt_predicted.numpy()))
        base_metrics.append(evaluate_predictions('DNN' + ' BASEDF', base_y_test, base_predicted.numpy()))

        samples_preds = tgt_test_df.groupby('sample_id').agg({'preds': lambda x: x.mode()[0]}).reset_index()
        samples_ytest = tgt_test_df.groupby('sample_id').agg({'y': lambda x: x.mode()[0]}).reset_index()

        metrics_grouped.append(
            evaluate_predictions('DNN' + 'Sample', samples_ytest['y'].tolist(),
                                    samples_preds['preds'].tolist()))
    
    # print("Metrics:\n", metrics_list, "\n grouped: \n", metrics_grouped, "\n base \n", base_metrics, "\n", n_tgt_train_samples)

    return metrics_list, metrics_grouped, base_metrics, n_tgt_train_samples