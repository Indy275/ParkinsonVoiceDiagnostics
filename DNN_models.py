import random
import pandas as pd
import numpy as np
import pandas as pd
from copy import deepcopy
import configparser
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from eval import evaluate_predictions
from data_util import get_samples


config = configparser.ConfigParser()
config.read('settings.ini')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')
k_folds = config.getint('EXPERIMENT_SETTINGS', 'kfolds')


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Define convolutional layers with dropout
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(2,2), stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.25)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(2,2), stride=1, padding=0)
        self.dropout2 = nn.Dropout(p=0.25)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2,2), stride=1, padding=0)
        self.dropout3 = nn.Dropout(p=0.25)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=1, padding=0)
        self.dropout4 = nn.Dropout(p=0.25)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(in_features=1792, out_features=256) 
        self.dropout5 = nn.Dropout(p=0.25)
        
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.dropout6 = nn.Dropout(p=0.25)
        
        self.fc3 = nn.Linear(in_features=128, out_features=2)
        
    def forward(self, x):
        # Pass input through convolutional layers with dropout and pooling
        # print(x.shape)
        x = self.pool1(F.relu(self.dropout1(self.conv1(x))))
        # print(x.shape)

        x = self.pool2(F.relu(self.dropout2(self.conv2(x))))
        # print(x.shape)

        x = self.pool3(F.relu(self.dropout3(self.conv3(x))))
        # print(x.shape)

        x = self.pool4(F.relu(self.dropout4(self.conv4(x))))
        # print(x.shape, "After Conv blocks")
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        # print(x.shape, "After flattening")
        
        # Pass through fully connected layers with dropout
        x = F.relu(self.dropout5(self.fc1(x)))
        # print(x.shape, "Lin1")

        x = F.relu(self.dropout6(self.fc2(x)))
        # print(x.shape)

        x = self.fc3(x)
        x = torch.sigmoid(x)
        # print(x.shape, "Output shape")
        
        return x
    
    
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

def create_dnn_model(n_features):
    input_size = n_features
    model = DNNModel(input_size)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, scheduler, criterion

def create_cnn_model():
    model = CNNModel()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, scheduler, criterion

def create_ResNet_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for params in model.parameters():
            params.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 2)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.6)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, scheduler, criterion


def train_model(model, optimizer, scheduler, criterion, train_loader, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(batch_data)  # Get model predictions
            # print(batch_data.shape, outputs.shape, batch_labels.shape)
            loss = criterion(outputs, batch_labels)  # Compute the loss
            # print(outputs, batch_labels, loss)
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model weights
            scheduler.step()

        if print_intermediate: # and epoch % (num_epochs-1) == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]: Loss: {loss.item():.4f}, lr:', round(optimizer.param_groups[0]['lr'],5))


def get_X_y(df, n_features, modeltype):
    if modeltype.startswith('DNNC'):
        df_grouped = df.groupby('sample_id')
        X = np.array([group.values for _, group in df_grouped])[:, :, :n_features].astype(float)
        X = torch.tensor(X).to(torch.float32).unsqueeze(1)
        X = X.repeat(1, 3, 1, 1)  # < ResNet
        y = torch.tensor(df_grouped['y'].first().values)
        df_new = df_grouped.first()
    else:
        X = torch.tensor(df.loc[:, df.columns[:n_features]].values).to(torch.float32)
        y = torch.tensor(df.loc[:, 'y'].values)
        df_new = df
    return df_new, X, y 

def run_dnn_model(modeltype, train_df, test_df):
    n_features = len(train_df.columns) - 5  # Ugly coding, but does the trick: all columns except last 4 are features

    # Get base train data
    train_df, X_train, y_train = get_X_y(train_df, n_features, modeltype)
    train_dataset = TensorDataset(X_train,y_train)
    train_loader = DataLoader(train_dataset, shuffle=True)

    # Create model
    if modeltype.startswith('DNNC'):
        # base_model, optimizer, scheduler, criterion = create_cnn_model()
        model, optimizer, scheduler, criterion = create_ResNet_model()
    else:
        model, optimizer, scheduler, criterion = create_dnn_model(n_features)

    # Train model on base train data
    train_model(model, optimizer, scheduler, criterion, train_loader)
    torch.save(model.state_dict(), 'DNNmodel')

    # Evaluation
    test_df, X_test, y_test = get_X_y(test_df, n_features, modeltype)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        _, predicted = torch.max(predictions.data, 1)
    test_df.loc[:, 'preds'] = predicted.numpy()

    all_metrics = evaluate_predictions('DNN', y_test, test_df)
    file_scores, subj_scores = zip(*all_metrics)

    return file_scores, subj_scores


def run_dnn_tl_model(scaler, modeltype, base_train_df, base_test_df, tgt_df):
    n_features = len(base_train_df.columns) - 5  # Ugly coding, but does the trick: all columns except last 4 are features

    # Get base train data
    base_train_df, base_X_train, base_y_train = get_X_y(base_train_df, n_features, modeltype)
    base_dataset = TensorDataset(base_X_train, base_y_train)
    base_loader = DataLoader(base_dataset, shuffle=True)

    # Create model
    if modeltype.startswith('DNNC'):
        base_model, optimizer, scheduler, criterion = create_cnn_model()
        # base_model, optimizer, scheduler, criterion = create_ResNet_model()
    else:
        base_model, optimizer, scheduler, criterion = create_dnn_model(n_features)

    # Train model on base train data
    train_model(base_model, optimizer, scheduler, criterion, base_loader)

    for params in base_model.parameters():
            params.requires_grad = False
    for p in [base_model.fc1,base_model.fc2,base_model.fc3]:
        for param in p.parameters():
            param.requires_grad = True

    metrics_list, metrics_grouped, base_metrics = [], [], []
    tgt_df_split = tgt_df.drop_duplicates(['subject_id'])
    tgt_df_split.loc[:,'ygender'] = tgt_df_split['y'].astype(str) + '_' + tgt_df_split['gender'].astype(str)
    
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    for i, (train_split_indices, test_split_indices) in enumerate(kf.split(tgt_df_split['subject_id'], tgt_df_split['ygender'])):
        print(f"Running subfold [{i+1}/{k_folds}]")
        scaler_copy = deepcopy(scaler)
        tgt_df_copy = deepcopy(tgt_df)
        model = deepcopy(base_model)
        model.load_state_dict(base_model.state_dict())
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        
        train_subjects = tgt_df_split.iloc[train_split_indices]['subject_id']
        test_subjects = tgt_df_split.iloc[test_split_indices]['subject_id']
        train_tgt_df = tgt_df_copy[tgt_df_copy['subject_id'].isin(train_subjects)]
        test_tgt_df = tgt_df_copy[tgt_df_copy['subject_id'].isin(test_subjects)]

        # Add target train data to scaler fit
        scaler_copy.partial_fit(train_tgt_df.iloc[:, :n_features].values) 
        train_tgt_df.iloc[:, :n_features] = scaler_copy.transform(train_tgt_df.iloc[:, :n_features].values)
        test_tgt_df.iloc[:, :n_features] = scaler_copy.transform(test_tgt_df.iloc[:, :n_features].values)
        
        tgt_train_df, tgt_X_train, tgt_y_train = get_X_y(train_tgt_df, n_features, modeltype)
        tgt_dataset = TensorDataset(tgt_X_train, tgt_y_train)
        tgt_loader = DataLoader(tgt_dataset, shuffle=True)
        
        # Fine-tune model
        train_model(model, optimizer, scheduler, criterion, tgt_loader, num_epochs=10)

        # Prepare test data
        tgt_test_df, tgt_X_test, tgt_y_test = get_X_y(test_tgt_df, n_features, modeltype)
        base_test_df, base_X_test, base_y_test = get_X_y(base_test_df, n_features, modeltype)

        # Evaluation
        model.eval()
        with torch.no_grad():
            base_preds = model(base_X_test)
            tgt_preds = model(tgt_X_test)
            _, base_predicted = torch.max(base_preds.data, 1)
            _, tgt_predicted = torch.max(tgt_preds.data, 1)
        tgt_test_df.loc[:, 'preds'] = tgt_predicted.numpy()
        base_test_df.loc[:, 'preds'] = base_predicted.numpy()

        all_metrics = evaluate_predictions(f'DNN_TL', tgt_y_test, tgt_test_df, base_y_test, base_test_df)
        metrics, grouped, base = zip(*all_metrics)
        metrics_list.append(metrics)
        metrics_grouped.append(grouped)
        base_metrics.append(base)
        
    return zip(*[metrics_list, metrics_grouped, base_metrics])


def run_dnn_fstl_model(scaler, modeltype, base_train_df, base_df, tgt_df):
    n_features = len(base_train_df.columns) - 5  # Ugly coding, but does the trick: all columns except last 4 are features

    # Get base train data
    base_train_df, base_X_train, base_y_train = get_X_y(base_train_df, n_features, modeltype)
    base_dataset = TensorDataset(base_X_train, base_y_train)
    base_loader = DataLoader(base_dataset, shuffle=True)

    # Create model
    if modeltype.startswith('DNNC'):
        base_model, optimizer, scheduler, criterion = create_cnn_model()
        base_model, optimizer, scheduler, criterion = create_ResNet_model()
    else:
        base_model, optimizer, scheduler, criterion = create_dnn_model(n_features)

    # Train model on base train data
    # train_model(base_model, optimizer, scheduler, criterion, base_loader)
    # base_model.load_state_dict(torch.load('DNNmodel', weights_only=True))

    # Prepare data for few-shot fine-tuning
    base_pos_subjs = list(base_df[base_df['y'] == 1]['subject_id'].unique())
    base_neg_subjs = list(base_df[base_df['y'] == 0]['subject_id'].unique())
    pos_subjs = list(tgt_df[tgt_df['y'] == 1]['subject_id'].unique())
    neg_subjs = list(tgt_df[tgt_df['y'] == 0]['subject_id'].unique())
    max_shot = min(len(pos_subjs), len(neg_subjs)) - 3  # Keep at least 3 pos and neg samples for evaluation
    
    metrics_list, metrics_grouped, base_metrics, n_tgt_train_samples = [], [], [], []
    seed = int(random.random()*10000)
    for n_shots in range(max_shot+1):
        scaler_copy = deepcopy(scaler)
        model = deepcopy(base_model)
        model.load_state_dict(base_model.state_dict())
        optimizer = optim.AdamW(model.parameters(), lr=0.0005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        if n_shots > 0:
            # Fine-tune model with pos and neg samples from base and target set
            base_train_df, base_test_df = get_samples(seed, base_pos_subjs, base_neg_subjs, max(1, int(n_shots)), base_df)
            tgt_train_df, tgt_test_df = get_samples(seed, pos_subjs, neg_subjs, n_shots, tgt_df)

            # Add target train data to scaler fit
            scaler_copy.partial_fit(tgt_train_df.iloc[:, :n_features].values) 
            tgt_train_df.iloc[:, :n_features] = scaler_copy.transform(tgt_train_df.iloc[:, :n_features].values).astype(float)
            tgt_test_df.iloc[:, :n_features] = scaler_copy.transform(tgt_test_df.iloc[:, :n_features].values).astype(float)

            tgt_train_df = pd.concat([tgt_train_df, base_train_df], ignore_index=True, axis=0)

            # Get target train data
            tgt_train_df, tgt_X_train, tgt_y_train = get_X_y(tgt_train_df, n_features, modeltype)
            tgt_dataset = TensorDataset(tgt_X_train, tgt_y_train)
            tgt_loader = DataLoader(tgt_dataset, shuffle=True)

            # Fine-tune model using target data
            train_model(model, optimizer, scheduler, criterion, tgt_loader, num_epochs=8)
        else: # n_shots == 0
            # Use entire tgt set for evaluation
            base_test_df = deepcopy(base_df)
            tgt_test_df = deepcopy(tgt_df)
            tgt_test_df.iloc[:, :n_features] = scaler_copy.transform(tgt_test_df.iloc[:, :n_features].values).astype(float)
        
        # Evaluation
        tgt_test_df, tgt_X_test, tgt_y_test = get_X_y(tgt_test_df, n_features, modeltype)
        base_test_df, base_X_test, base_y_test = get_X_y(base_test_df, n_features, modeltype)

        model.eval()
        with torch.no_grad():
            base_preds = model(base_X_test)
            tgt_preds = model(tgt_X_test)
            _, base_predicted = torch.max(base_preds.data, 1)
            _, tgt_predicted = torch.max(tgt_preds.data, 1)
        tgt_test_df.loc[:, 'preds'] = tgt_predicted.numpy()
        base_test_df.loc[:, 'preds'] = base_predicted.numpy()


        all_metrics = evaluate_predictions(f'DNN ({n_shots} shots)', tgt_y_test, tgt_test_df, base_y_test, base_test_df)
        metrics, grouped, base = zip(*all_metrics)
        metrics_list.append(metrics)
        metrics_grouped.append(grouped)
        base_metrics.append(base)
        n_tgt_train_samples.append(n_shots)
        
    return zip(*[metrics_list, metrics_grouped, base_metrics, n_tgt_train_samples])