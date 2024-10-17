import random
import os
from copy import deepcopy
import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from eval import evaluate_predictions

config = configparser.ConfigParser()
config.read('settings.ini')
plot_fimp = config.getboolean('OUTPUT_SETTINGS', 'plot_fimp')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')


def run_ml_model(X_train, X_test, y_train, y_test, df, test_indices):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    df.loc[test_indices, 'preds'] = preds

    file_scores = evaluate_predictions('RFC' + 'FileID', y_test.tolist(), df.loc[test_indices, 'preds'].tolist())

    # Scores aggregated per speaker
    samples_preds = df.loc[test_indices, ['subject_id', 'preds']].groupby('subject_id').agg(
        {'preds': lambda x: x.mode()[0]}).reset_index()
    samples_ytest = df.loc[test_indices, ['subject_id', 'y']].groupby('subject_id').agg(
        {'y': lambda x: x.mode()[0]}).reset_index()
    subj_scores = evaluate_predictions('RFC' + 'SubjID', samples_ytest['y'].tolist(), samples_preds['preds'].tolist())

    if plot_fimp:
        fimp = sorted(zip(df.columns[:X_train.shape[1]], clf.feature_importances_), key=lambda l: l[1], reverse=True)
        # [6, 5, 6, 156]
        fl = [0, 6, 11, 17, 20, 176]
        f0_features = [str(x) for x in range(fl[0], fl[1])]
        formant_features = [str(x) for x in range(fl[1], fl[2])]
        jitter_features = [str(x) for x in range(fl[2], fl[3])]
        shimmer_features = [str(x) for x in range(fl[3], fl[4])]
        mfcc_features = [str(x) for x in range(fl[4], fl[5])]

        sum_f0 = sum(val for key, val in fimp if key in f0_features)
        print("Contribution of F0:", sum_f0)
        sum_form = sum(val for key, val in fimp if key in formant_features)
        print("Contribution of formants:", sum_form)
        sum_jit = sum(val for key, val in fimp if key in jitter_features)
        print("Contribution of Jitter:", sum_jit)
        sum_shim = sum(val for key, val in fimp if key in shimmer_features)
        print("Contribution of Shimmer:", sum_shim)
        sum_mfcc = sum(val for key, val in fimp if key in mfcc_features)
        print("Contribution of MFCC:", sum_mfcc)
        print("Total feature importance (should equal to 1):", sum_mfcc + sum_f0 + sum_form + sum_jit + sum_shim)

        # fimp_mfcc = [(key, val) for key, val in fimp if key in mfcc_features]
        plt.barh(df.columns[fl[0]:fl[1]], clf.feature_importances_[fl[0]:fl[1]], color='green')
        plt.barh(df.columns[fl[1]:fl[2]], clf.feature_importances_[fl[1]:fl[2]], color='blue')
        plt.barh(df.columns[fl[2]:fl[3]], clf.feature_importances_[fl[2]:fl[3]], color='red')
        plt.barh(df.columns[fl[3]:fl[4]], clf.feature_importances_[fl[3]:fl[4]], color='purple')
        # plt.barh(df.columns[fl[4]:fl[5]], clf.feature_importances_[fl[4]:fl[5]], color='orange')

        plt.barh(df.columns[fl[4]:fl[4]+39], clf.feature_importances_[fl[4]:fl[4]+39], color='green')
        plt.barh(df.columns[fl[4]+39:fl[4]+39+39], clf.feature_importances_[fl[4]+39:fl[4]+39+39], color='blue')
        plt.barh(df.columns[fl[4]+39+39:fl[4]+78+39], clf.feature_importances_[fl[4]+39+39:fl[4]+78+39], color='red')
        plt.barh(df.columns[fl[4]+78+39:fl[4]+78+78], clf.feature_importances_[fl[4]+78+39:fl[4]+78+78], color='orange')

        plt.yticks(
            [(fl[0] + fl[1]) / 2, (fl[1] + fl[2]) / 2, (fl[2] + fl[3]) / 2, (fl[3] + fl[4]) / 2, (fl[4] + fl[5]) / 2],
            ['F0', 'Jitter', 'Shimmer', 'Formants', 'MFCC'])
        plt.xlabel("Relative feature importance")
        plt.tight_layout()
        plt.ylim((fl[0], fl[-1]))
        plt.show()

    return file_scores, subj_scores


def train_model(model, X, y):
    model.fit(X, y)

def run_ml_tl_model(base_X_train, base_X_test, base_y_train, base_y_test, tgt_df):
    n_features = np.shape(base_X_train)[1]
    clf = RandomForestClassifier()
    train_model(clf, base_X_train, base_y_train)

    pos_subjs = list(tgt_df[tgt_df['y'] == 1]['subject_id'].unique())
    neg_subjs = list(tgt_df[tgt_df['y'] == 0]['subject_id'].unique())
    max_shot = min(len(pos_subjs), len(neg_subjs)) - 5

    metrics_list, metrics_grouped, n_tgt_train_samples, base_metrics = [], [], [], []
    seed = int(random.random()*10000)
    for n_shots in range(max_shot):
        clf_copy = deepcopy(clf)  # Copy model trained on base language

        random.seed(seed)
        pos_train_samples = random.sample(pos_subjs, n_shots)
        random.seed(seed)
        neg_train_samples = random.sample(neg_subjs, n_shots)

        tgt_train_df = tgt_df[tgt_df['subject_id'].isin(pos_train_samples + neg_train_samples)]
        tgt_test_df = tgt_df[~tgt_df['subject_id'].isin(pos_train_samples + neg_train_samples)]

        if n_shots > 0:
            # Train model with a mix of base and target samples
            tgt_X_train = tgt_train_df.iloc[:, :n_features]
            tgt_y_train = tgt_train_df['y']
            train_model(clf_copy, tgt_X_train, tgt_y_train)

        tgt_X_test = tgt_test_df.iloc[:, :n_features]
        tgt_y_test = tgt_test_df['y']

        predictions = clf_copy.predict(tgt_X_test)
        tgt_test_df.loc[:, 'preds'] = predictions

        metrics_list.append(evaluate_predictions('RFC' + '0{}shot'.format(n_shots), tgt_y_test, predictions))
        n_tgt_train_samples.append(n_shots)

        base_metrics.append(evaluate_predictions('RFC' + ' BASEDF', base_y_test, clf_copy.predict(base_X_test)))

        samples_preds = tgt_test_df.groupby('sample_id').agg({'preds': lambda x: x.mode()[0]}).reset_index()
        samples_ytest = tgt_test_df.groupby('sample_id').agg({'y': lambda x: x.mode()[0]}).reset_index()

        metrics_grouped.append(
            evaluate_predictions('RFC' + 'Sample', samples_ytest['y'].tolist(),
                                    samples_preds['preds'].tolist()))
    
    # print("Metrics:\n", metrics_list, "\n grouped: \n", metrics_grouped, "\n base \n", base_metrics, "\n", n_tgt_train_samples)

    return metrics_list, metrics_grouped, base_metrics, n_tgt_train_samples