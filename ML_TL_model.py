import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import random
from copy import deepcopy

from data_util.data_util import load_data, split_data, old_load_data
from eval import evaluate_predictions

import configparser

config = configparser.ConfigParser()
config.read('settings.ini')

fine_tune_size = [int(x) for x in config['EXPERIMENT_SETTINGS']['fine_tune_size'].split(",")]


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)


def run_experiment(base_dataset, target_dataset, ifm_nifm):
    base_df, train_features = load_data(base_dataset, ifm_nifm)
    target_df, target_features = load_data(target_dataset, ifm_nifm)
    assert train_features == target_features, "Number of features across languages should be equal: {} and {}".format(
        train_features, target_features)

    X_train = base_df.iloc[:, :train_features]
    y_train = base_df['y']

    print("X_train:", X_train.shape, "; y_train: ", y_train.shape, "; target_df: ", target_df.shape)
    rfc = RandomForestClassifier(random_state=11)
    classifiers = [rfc]
    clf_names = ['Random Forest Classifier']  # , 'XGBoost Classifier']
    for clf, clf_name in zip(classifiers, clf_names):
        print("Fitting {} to train data".format(clf_name))
        train_model(clf, X_train, y_train)
        # evaluate_predictions(clf_name + ' TRAIN', y_train, clf.predict(X_train))

        metrics_list, metrics_grouped, n_tgt_train_samples = [], [], []
        for n_shots in fine_tune_size:
            clf_copy = deepcopy(clf)  # Copy model trained on base language

            pos_subjs = list(target_df[target_df['y'] == 1]['subject_id'].unique())
            neg_subjs = list(target_df[target_df['y'] == 0]['subject_id'].unique())

            if n_shots > min(len(pos_subjs), len(neg_subjs))-5:
                n_shots = min(len(pos_subjs), len(neg_subjs)) - 5  # -5 to ensure at least 5 test samples

            random.seed(1)
            pos_train_samples = random.sample(pos_subjs, n_shots)
            random.seed(1)
            neg_train_samples = random.sample(neg_subjs, n_shots)
            target_train_df = target_df[target_df['subject_id'].isin(pos_train_samples + neg_train_samples)]
            target_test_df = target_df[~target_df['subject_id'].isin(pos_train_samples + neg_train_samples)]

            if n_shots > 0:
                # Train model with these additional samples
                X_target_train = target_train_df.iloc[:, :target_features]
                y_target_train = target_train_df['y']
                train_model(clf_copy, X_target_train, y_target_train)

            X_target_test = target_test_df.iloc[:, :target_features]
            y_target_test = target_test_df['y']
            predictions = clf_copy.predict(X_target_test)
            target_test_df.loc[:, 'preds'] = predictions

            metrics_list.append(evaluate_predictions(clf_name + '0{}shot'.format(n_shots),y_target_test, predictions))
            n_tgt_train_samples.append(n_shots)

            if ifm_nifm[-4] == 'n':  # wiNdow or Nifm; majority voting only sensible for window-level predictions
                samples_preds = target_test_df.groupby('sample_id').agg({'preds': lambda x: x.mode()[0]}).reset_index()
                samples_ytest = target_test_df.groupby('sample_id').agg({'y': lambda x: x.mode()[0]}).reset_index()

                metrics_grouped.append(
                    evaluate_predictions(clf_name + 'Sample', samples_ytest['y'].tolist(), samples_preds['preds'].tolist()))

        metrics_df = pd.DataFrame(metrics_list, columns=['Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity'])
        metrics_df['Iteration'] = n_tgt_train_samples
        metrics_df.to_csv('experiments\\metrics_{}_{}.csv'.format(base_dataset, target_dataset), index=False)

        if ifm_nifm[-4] == 'n':  # wiNdow or Nifm; majority voting only sensible for window-level predictions
            metrics_grouped = pd.DataFrame(metrics_list, columns=['Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity'])
            metrics_grouped['Iteration'] = n_tgt_train_samples
            metrics_grouped.to_csv('experiments\\metrics_{}_{}_grouped.csv'.format(base_dataset, target_dataset), index=False)


