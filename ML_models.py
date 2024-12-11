import random
from joblib import dump, load
from copy import deepcopy
import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from eval import evaluate_predictions

from data_util import get_samples

config = configparser.ConfigParser()
config.read('settings.ini')
plot_fimp = config.getboolean('OUTPUT_SETTINGS', 'plot_fimp')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')

if plot_fimp:
    from plotting.results_visualised import fimp_plot, fimp_plot_nifm

def get_X_y(df):
    n_features = len(df.columns) - 5  # Ugly coding, but does the trick: all columns except last 4 are features
    X = df.loc[:, df.columns[:n_features]].values
    y = df.loc[:, 'y'].values
    return df, X, y, n_features


def run_ml_model(train_df, test_df):
    # Get base train data
    train_df, X_train, y_train, n_features = get_X_y(train_df)

    # Create model
    # clf = SVC(kernel='linear')
    clf = SGDClassifier()
    # clf = RandomForestClassifier()

    # Train model on base train data
    clf.fit(X_train, y_train)

    # Evaluation
    test_df, X_test, y_test, _ = get_X_y(test_df)

    predicted = clf.predict(X_test)
    test_df.loc[:, 'preds'] = predicted

    all_metrics = evaluate_predictions('SVM', y_test, test_df)
    file_scores, subj_scores = zip(*all_metrics)

    if plot_fimp:
        # fimp = sorted(zip(test_df.columns[:X_test.shape[1]], clf.feature_importances_), key=lambda l: l[1], reverse=True)
        # fimp = clf.feature_importances_
        # fimp = list(map(np.abs, clf.coef_[0]))
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        fimp = clf.feature_importances_
        return file_scores, subj_scores, fimp

        fimp_plot(fimp, test_df)
        return file_scores, subj_scores, fimp
    return file_scores, subj_scores


def run_ml_tl_model(scaler, base_train_df, base_test_df, tgt_df):
    # Get base train data
    base_train_df, base_X_train, base_y_train, n_features = get_X_y(base_train_df)
    base_X_train[np.isnan(base_X_train)] = 0

    # Create model and train model on base train data
    base_clf = SGDClassifier()
    base_clf.partial_fit(base_X_train, base_y_train, classes=np.unique(base_y_train))

    # Prepare data for fine-tuning
    base_train_df.reset_index(drop=True, inplace=True) 
    base_test_df.reset_index(drop=True, inplace=True) 
    base_df = pd.concat([base_train_df, base_test_df])
    base_pos_subjs = list(base_df[base_df['y'] == 1]['subject_id'].unique())
    base_neg_subjs = list(base_df[base_df['y'] == 0]['subject_id'].unique())

    pos_subjs = list(tgt_df[tgt_df['y'] == 1]['subject_id'].unique())
    neg_subjs = list(tgt_df[tgt_df['y'] == 0]['subject_id'].unique())

    metrics_list, metrics_grouped, base_metrics = [], [], []
    seed = int(random.random()*10000)
    scaler_copy = deepcopy(scaler)
    clf = deepcopy(base_clf)  # Copy model trained on base language
    n_tgt_samples = min(len(pos_subjs), len(neg_subjs)) - 3  # All but 3 pos and neg samples to finetune model
    tgt_train_df, tgt_test_df = get_samples(seed, pos_subjs, neg_subjs, n_tgt_samples, tgt_df)

    # Add target train data to scaler fit
    scaler_copy.partial_fit(tgt_train_df.iloc[:, :n_features].values) 
    tgt_train_df.iloc[:, :n_features] = scaler_copy.transform(tgt_train_df.iloc[:, :n_features].values)
    tgt_test_df.iloc[:, :n_features] = scaler_copy.transform(tgt_test_df.iloc[:, :n_features].values)

    # Concatenate train data
    tgt_train_df.reset_index(drop=True, inplace=True) 
    base_train_df.reset_index(drop=True, inplace=True) 
    tgt_train_df = pd.concat([tgt_train_df, base_train_df])

    tgt_train_df, tgt_X_train, tgt_y_train, _ = get_X_y(tgt_train_df)
    tgt_X_train[np.isnan(tgt_X_train)] = 0

    # Fine-tuning
    clf.partial_fit(tgt_X_train, tgt_y_train)
    # clf.fit(tgt_X_train, tgt_y_train)

    # Prepare test data
    tgt_test_df, tgt_X_test, tgt_y_test, _ = get_X_y(tgt_test_df)
    base_test_df, base_X_test, base_y_test, _ = get_X_y(base_test_df)

    # Evaluation
    base_preds = clf.predict(base_X_test)
    tgt_preds = clf.predict(tgt_X_test)
    base_test_df.loc[:, 'preds'] = base_preds
    tgt_test_df.loc[:, 'preds'] = tgt_preds

    all_metrics = evaluate_predictions(f'SVM', tgt_y_test, tgt_test_df, base_y_test, base_test_df)
    metrics, grouped, base = zip(*all_metrics)
    metrics_list.append(metrics)
    metrics_grouped.append(grouped)
    base_metrics.append(base)

    if plot_fimp:
        # fimp = sorted(zip(test_df.columns[:X_test.shape[1]], clf.feature_importances_), key=lambda l: l[1], reverse=True)
        # fimp = clf.feature_importances_
        fimp = list(map(np.abs, clf.coef_[0]))
        fimp_plot(fimp, tgt_test_df)
    
    return zip(*[metrics_list, metrics_grouped, base_metrics])


def run_ml_fstl_model(scaler, base_train_df, base_test_df, tgt_df):
    # Get base train data
    base_train_df, base_X_train, base_y_train, n_features = get_X_y(base_train_df)

    #Create model and train model on base train data
    base_clf = SGDClassifier()
    base_clf.partial_fit(base_X_train, base_y_train, classes=np.unique(base_y_train))

    # Prepare data for few-shot fine-tuning
    base_train_df.reset_index(drop=True, inplace=True) 
    base_test_df.reset_index(drop=True, inplace=True) 
    base_df = pd.concat([base_train_df, base_test_df])
    base_pos_subjs = list(base_df[base_df['y'] == 1]['subject_id'].unique())
    base_neg_subjs = list(base_df[base_df['y'] == 0]['subject_id'].unique())

    pos_subjs = list(tgt_df[tgt_df['y'] == 1]['subject_id'].unique())
    neg_subjs = list(tgt_df[tgt_df['y'] == 0]['subject_id'].unique())
    max_shot = min(len(pos_subjs), len(neg_subjs)) - 1  # Keep at least 3 pos and neg samples for evaluation

    metrics_list, metrics_grouped, n_tgt_train_samples, base_metrics = [], [], [], []
    seed = int(random.random()*10000)
    for n_shots in range(0, max_shot+1):
        base_test_df_copy = deepcopy(base_test_df)
        tgt_test_df = deepcopy(tgt_df)
        scaler_copy = deepcopy(scaler)
        clf = deepcopy(base_clf)  # Copy model trained on base language

        if n_shots > 0:
            # Fine-tune model with pos and neg samples from base and target set
            base_train_df, base_test_df_copy = get_samples(seed, base_pos_subjs, base_neg_subjs, int(n_shots/4), base_df)
            tgt_train_df, tgt_test_df = get_samples(seed, pos_subjs, neg_subjs, n_shots, tgt_df)

            # Add target train data to scaler fit
            scaler_copy.partial_fit(tgt_train_df.iloc[:, :n_features].values) 
            tgt_train_df.iloc[:, :n_features] = scaler_copy.transform(tgt_train_df.iloc[:, :n_features].values)

            # Concatenate train data
            # tgt_train_df = pd.concat([tgt_train_df, base_train_df])
            
            # Get target train data
            tgt_train_df, tgt_X_train, tgt_y_train, _ = get_X_y(tgt_train_df)
            
            # Fine-tune model using target data
            # clf.fit(tgt_X_train, tgt_y_train)
            clf.partial_fit(tgt_X_train, tgt_y_train)

        # Prepare data for evaluation
        tgt_test_df.iloc[:, :n_features] = scaler_copy.transform(tgt_test_df.iloc[:, :n_features].values)
        tgt_test_df, tgt_X_test, tgt_y_test, _ = get_X_y(tgt_test_df)
        base_test_df_copy, base_X_test, base_y_test, _ = get_X_y(base_test_df_copy)
        
        # Evaluate model
        base_preds = clf.predict(base_X_test)
        tgt_preds = clf.predict(tgt_X_test)

        base_test_df_copy.loc[:, 'preds'] = base_preds
        tgt_test_df.loc[:, 'preds'] = tgt_preds

        all_metrics = evaluate_predictions(f'SVM ({n_shots} shots)', tgt_y_test, tgt_test_df, base_y_test, base_test_df_copy)
        metrics, grouped, base = zip(*all_metrics)
        metrics_list.append(metrics)
        metrics_grouped.append(grouped)
        base_metrics.append(base)
        n_tgt_train_samples.append(n_shots)

    return zip(*[metrics_list, metrics_grouped, base_metrics, n_tgt_train_samples])