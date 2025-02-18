from copy import deepcopy
import numpy as np
import configparser
from sklearn.linear_model import SGDClassifier

config = configparser.ConfigParser()
config.read('settings.ini')
plot_fimp = config.getboolean('OUTPUT_SETTINGS', 'plot_fimp')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')

class SGD_model:
    def __init__(self, mono):
        self.name = 'SGD'
        self.mono = mono
        self.base_model = None

    def copy(self):
        self.model = deepcopy(self.base_model)

    def create_model(self, n_features):
        self.model = SGDClassifier()
        
    def train(self, train_loader):
        X_train, y_train = train_loader
        if self.mono:
            self.model.fit(X_train, y_train)
        else:  # cross-lingual
            self.model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        if self.base_model == None:  # base model training -> save model state
            self.base_model = self.model
    
    
    def eval_monolingual(self, X_test):
        preds = self.model.predict_proba(X_test)
        return preds

    def eval_multilingual(self, base_X_test, tgt_X_test):
        base_preds = self.model.predict(base_X_test)
        tgt_preds = self.model.predict(tgt_X_test)
        return base_preds, tgt_preds
        
    def get_X_y(self, df, train=False):
        n_features = len(df.columns) - 5  # Ugly coding, but does the trick: all columns except last 4 are features
        X = df.loc[:, df.columns[:n_features]].values
        # X[np.isnan(X)] = 0
        y = df.loc[:, 'y'].values
        if train:
            return df, (X, y), n_features
        return df, X, y
