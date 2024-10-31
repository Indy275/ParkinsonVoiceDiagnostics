import random
import numpy as np
import pandas as pd
from copy import deepcopy
import configparser
from data_util import get_samples
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from eval import evaluate_predictions

config = configparser.ConfigParser()
config.read('settings.ini')
plot_fimp = config.getboolean('OUTPUT_SETTINGS', 'plot_fimp')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')


class DNNModel(nn.Module):
    def __init__(self, input_size, dropout_prob=0.25):
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
        x = torch.sigmoid(x)

        return x


def train_model(model, optimizer, criterion, train_loader, num_epochs=10):
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


def run_dnn_model(X_train, X_test, y_train, y_test, test_df):
    n_features = X_train.shape[1]

    X_train = torch.tensor(X_train.values).to(torch.float32)
    X_test = torch.tensor(X_test.values).to(torch.float32)
    y_train = torch.tensor(y_train.values)
    y_test = y_test.values
    model, optimizer, criterion = create_dnn_model(n_features)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    train_model(model, optimizer, criterion, train_loader)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        _, predicted = torch.max(predictions.data, 1)
    test_df['preds'] = predicted.numpy()

    all_metrics = evaluate_predictions('DNN', y_test, test_df)
    file_scores, subj_scores = zip(*all_metrics)

    return file_scores, subj_scores


def train_tl_batch_model(model, optimizer, criterion, base_loader, target_loader, num_epochs=10):
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


def train_tl_model(model, optimizer, criterion, base_loader, target_loader=None, num_epochs=10):
       for epoch in range(num_epochs):
        model.train()
        
        for batch_data, batch_labels in base_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(batch_data)  # Get model predictions
            loss = criterion(outputs, batch_labels)  # Compute the loss
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model weights

        if print_intermediate and epoch % (num_epochs-1) == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]: Loss: {loss.item():.4f}')


def run_dnn_tl_model(scaler, base_X_train, base_X_test, base_y_train, base_y_test, base_df, tgt_df):
    n_features = base_X_train.shape[1]

    base_X_train = torch.tensor(base_X_train).to(torch.float32)
    base_X_test = torch.tensor(base_X_test).to(torch.float32)
    base_y_train = torch.tensor(base_y_train)
    base_y_test = base_y_test
    base_model, optimizer, criterion = create_dnn_model(n_features)

    base_dataset = TensorDataset(base_X_train, base_y_train)
    base_loader = DataLoader(base_dataset)#, batch_size=5, shuffle=True)

    train_tl_model(base_model, optimizer, criterion, base_loader, target_loader=None)

    base_pos_subjs = list(base_df[base_df['y'] == 1]['subject_id'].unique())
    base_neg_subjs = list(base_df[base_df['y'] == 0]['subject_id'].unique())

    pos_subjs = list(tgt_df[tgt_df['y'] == 1]['subject_id'].unique())
    neg_subjs = list(tgt_df[tgt_df['y'] == 0]['subject_id'].unique())
    max_shot = min(len(pos_subjs), len(neg_subjs)) - 5
    max_shot = 20
    
    metrics_list, metrics_grouped, base_metrics, n_tgt_train_samples = [], [], [], []
    seed = int(random.random()*10000)
    for n_shots in range(max_shot+1):
        scaler_copy = deepcopy(scaler)
        model = deepcopy(base_model)
        model.load_state_dict(base_model.state_dict())
        optimizer = optim.AdamW(model.parameters(), lr=0.0005)

        if n_shots > 0:
            # Fine-tune model with pos and neg samples from base and target set
            base_train_df, _ = get_samples(seed, base_pos_subjs, base_neg_subjs, max(1, int(n_shots/4)), base_df)
            tgt_train_df, tgt_test_df = get_samples(seed, pos_subjs, neg_subjs, n_shots, tgt_df)

            # Add target train data to scaler fit
            scaler_copy.partial_fit(tgt_train_df.iloc[:, :n_features].values) 

            # Transform target train and test data
            tgt_train_df.iloc[:, :n_features] = scaler_copy.transform(tgt_train_df.iloc[:, :n_features].values)
            tgt_test_df.iloc[:, :n_features] = scaler_copy.transform(tgt_test_df.iloc[:, :n_features].values)

            tgt_train_df = pd.concat([tgt_train_df, base_train_df])

            tgt_X_train = tgt_train_df.iloc[:, :n_features].values
            tgt_X_train = torch.tensor(tgt_X_train).to(torch.float32)
            tgt_y_train = torch.tensor(tgt_train_df['y'].values)

            tgt_dataset = TensorDataset(tgt_X_train, tgt_y_train)
            tgt_loader = DataLoader(tgt_dataset)#, batch_size=5, shuffle=True)

            # train_tl_batch_model(model, optimizer, criterion, base_loader, tgt_loader)
            train_tl_model(model, optimizer, criterion, tgt_loader, num_epochs=5)
        elif n_shots == 0:
            # Use entire tgt set for evaluation
            tgt_test_df = deepcopy(tgt_df)
            tgt_test_df.iloc[:, :n_features] = scaler_copy.transform(tgt_test_df.iloc[:, :n_features].values)

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

        all_metrics = evaluate_predictions(f'DNN ({n_shots} shots)', tgt_y_test, tgt_test_df, base_y_test, base_predicted.numpy())
        metrics, grouped, base = zip(*all_metrics)
        metrics_list.append(metrics)
        metrics_grouped.append(grouped)
        base_metrics.append(base)
        n_tgt_train_samples.append(n_shots)
        
    return metrics_list, metrics_grouped, base_metrics, n_tgt_train_samples