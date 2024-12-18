import random
import pandas as pd
import numpy as np
import pandas as pd
from copy import deepcopy
import configparser
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

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

class BaseModel:
    def __init__(self):
        self.base_model = None
        pass

    def copy(self):
        self.model = deepcopy(self.base_model)
        self.model.load_state_dict(self.model.state_dict())
        return self.model

    def train(self, train_loader, num_epochs=5):
        for epoch in range(num_epochs):
            self.model.train()
            train_acc = []
            for batch_data, batch_labels in train_loader:
                self.optimizer.zero_grad()  # Reset gradients

                outputs = self.model(batch_data)  # Get model predictions
                _, prediction = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, batch_labels)  # Compute the loss
                train_acc.append(accuracy_score(prediction.numpy(), batch_labels))
                loss.backward()  # Compute gradients
                self.optimizer.step()  # Update model weights
                self.scheduler.step()

            if print_intermediate: # and epoch % (num_epochs-1) == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]: loss: {loss.item():.4f}, lr:', round(self.optimizer.param_groups[0]['lr'],5),
                    ', train acc:', round(np.mean(train_acc), 3))
        
        if self.base_model == None:  # base model training -> save model state
            self.base_model = self.model
    
    
    def get_X_y(self, df, train=False):
        n_features = len(df.columns) - 5  # Ugly coding, but does the trick: all columns except last 4 are features
        if self.name == 'CNN' or self.name == 'ResNet':
            df_grouped = df.groupby('sample_id')
            X = np.array([group.values for _, group in df_grouped])[:, :, :n_features].astype(float)
            X = torch.tensor(X).to(torch.float32).unsqueeze(1)
            y = torch.tensor(df_grouped['y'].first().values)
            df = df_grouped.first()
            if self.name == 'ResNet':
                X = X.repeat(1, 3, 1, 1)  # ResNet requires 3 dimensions: RGB
        else:
            X = torch.tensor(df.loc[:, df.columns[:n_features]].values).to(torch.float32)
            y = torch.tensor(df.loc[:, 'y'].values)
        
        tensor_dataset = TensorDataset(X,y)
        data_loader = DataLoader(tensor_dataset, shuffle=True)
        if train:
            return df, data_loader, n_features
        return df, X, y


    def eval_monolingual(self, X_test):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test)
            _, predicted = torch.max(preds.data, 1)
        return predicted.numpy()
    

    def eval_multilingual(self, base_X_test, tgt_X_test):
        self.model.eval()
        with torch.no_grad():
            base_preds = self.model(base_X_test)
            tgt_preds = self.model(tgt_X_test)
            _, base_predicted = torch.max(base_preds.data, 1)
            _, tgt_predicted = torch.max(tgt_preds.data, 1)
        return base_predicted.numpy(), tgt_predicted.numpy()

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
    

class CNN_model(BaseModel):
    def __init__(self, mono):
        super().__init__()
        self.name = 'CNN'
        self.mono = mono
    
    def create_model(self, **kwargs):
        self.model = CNNModel()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()


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
    

class DNN_model(BaseModel):
    def __init__(self, mono):
        super().__init__()
        self.name = 'DNN'
        self.mono = mono

    def create_model(self, n_features, **kwargs):
        self.model = DNNModel(input_size=n_features)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()


class PT_model(BaseModel):
    def __init__(self, mono):
        super().__init__()
        self.name = 'PTM'
        self.mono = mono

    def create_model(self, **kwargs):
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.classifier = DNNModel(input_size=128)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.00005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, num_epochs=5):
        for epoch in range(num_epochs):
            self.model.train()
            train_acc = []
            for batch_data, batch_labels in train_loader:
                self.optimizer.zero_grad()  # Reset gradients
                # batch_data = np.squeeze(batch_data.numpy())  # <- PTM

                outputs = self.model(batch_data)  # Get model predictions
                # outputs = self.model.classifier(outputs)  # <- PTM
                # batch_labels = batch_labels.repeat(outputs.shape[0])  # <- PTM

                _, prediction = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, batch_labels)  # Compute the loss
                train_acc.append(accuracy_score(prediction.numpy(), batch_labels))
                loss.backward()  # Compute gradients
                self.optimizer.step()  # Update model weights
                self.scheduler.step()

            if print_intermediate: # and epoch % (num_epochs-1) == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]: loss: {loss.item():.4f}, lr:', round(self.optimizer.param_groups[0]['lr'],5),
                    ', train acc:', round(np.mean(train_acc), 3))


class ResNet_model(BaseModel):
    def __init__(self, mono):
        super().__init__()
        self.name = 'CNN'
        self.mono = mono

    def create_model(self, **kwargs):
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for params in self.model.parameters():
            params.requires_grad = True
        # for params in model.parameters():
        #         params.requires_grad = False
        # for param in model.layer4.parameters():
        #     param.requires_grad = True
        # for param in model.fc.parameters():
        #     param.requires_grad = True

        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.6)
        self.criterion = nn.CrossEntropyLoss()
