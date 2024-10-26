import random
from joblib import dump, load
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


def run_ml_model(X_train, X_test, y_train, y_test, test_df):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    test_df['preds'] = preds

    all_metrics = evaluate_predictions('RFC', y_test, test_df)
    file_scores, subj_scores = zip(*all_metrics)

    dump(clf, 'RFCmodel.pkl')

    if plot_fimp:
        fimp = sorted(zip(test_df.columns[:X_test.shape[1]], clf.feature_importances_), key=lambda l: l[1], reverse=True)
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
        plt.barh(test_df.columns[fl[0]:fl[1]], clf.feature_importances_[fl[0]:fl[1]], color='green')
        plt.barh(test_df.columns[fl[1]:fl[2]], clf.feature_importances_[fl[1]:fl[2]], color='blue')
        plt.barh(test_df.columns[fl[2]:fl[3]], clf.feature_importances_[fl[2]:fl[3]], color='red')
        plt.barh(test_df.columns[fl[3]:fl[4]], clf.feature_importances_[fl[3]:fl[4]], color='purple')
        # plt.barh(df.columns[fl[4]:fl[5]], clf.feature_importances_[fl[4]:fl[5]], color='orange')

        plt.barh(test_df.columns[fl[4]:fl[4]+39], clf.feature_importances_[fl[4]:fl[4]+39], color='green')
        plt.barh(test_df.columns[fl[4]+39:fl[4]+39+39], clf.feature_importances_[fl[4]+39:fl[4]+39+39], color='blue')
        plt.barh(test_df.columns[fl[4]+39+39:fl[4]+78+39], clf.feature_importances_[fl[4]+39+39:fl[4]+78+39], color='red')
        plt.barh(test_df.columns[fl[4]+78+39:fl[4]+78+78], clf.feature_importances_[fl[4]+78+39:fl[4]+78+78], color='orange')

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

def run_ml_tl_model(scaler, base_X_train, base_X_test, base_y_train, base_y_test, tgt_df):
    n_features = np.shape(base_X_train)[1]
    clf = RandomForestClassifier()
    train_model(clf, base_X_train, base_y_train)

    pos_subjs = list(tgt_df[tgt_df['y'] == 1]['subject_id'].unique())
    neg_subjs = list(tgt_df[tgt_df['y'] == 0]['subject_id'].unique())
    max_shot = min(len(pos_subjs), len(neg_subjs)) - 5

    metrics_list, metrics_grouped, n_tgt_train_samples, base_metrics = [], [], [], []
    seed = int(random.random()*10000)
    for n_shots in range(max_shot+1):
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
            tgt_X_train = scaler.transform(tgt_X_train.values)  # Uncertain if this is correct way
            train_model(clf_copy, tgt_X_train, tgt_y_train)

        tgt_X_test = tgt_test_df.iloc[:, :n_features]
        tgt_X_test = scaler.transform(tgt_X_test.values)  
        tgt_y_test = tgt_test_df['y']

        base_preds = clf_copy.predict(base_X_test)
        tgt_preds = clf_copy.predict(tgt_X_test)
        tgt_test_df.loc[:, 'preds'] = tgt_preds
        all_metrics = evaluate_predictions(f'RFC ({n_shots} shots)', tgt_y_test, tgt_test_df, base_y_test, base_preds)
        metrics, grouped, base = zip(*all_metrics)
        metrics_list.append(metrics)
        metrics_grouped.append(grouped)
        base_metrics.append(base)
        n_tgt_train_samples.append(n_shots)
    
    # print("Metrics:\n", metrics_list, "\n grouped: \n", metrics_grouped, "\n base \n", base_metrics, "\n", n_tgt_train_samples)

    return metrics_list, metrics_grouped, base_metrics, n_tgt_train_samples