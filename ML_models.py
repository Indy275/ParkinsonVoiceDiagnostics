import random
from joblib import dump, load
from copy import deepcopy
import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt

# from skmultiflow.meta import AdaptiveRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from eval import evaluate_predictions

from data_util import get_samples

config = configparser.ConfigParser()
config.read('settings.ini')
plot_fimp = config.getboolean('OUTPUT_SETTINGS', 'plot_fimp')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')


def run_ml_model(X_train, X_test, y_train, y_test, test_df):
    # clf = RandomForestClassifier()
    clf = SVC()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    test_df['preds'] = preds

    all_metrics = evaluate_predictions('RFC', y_test, test_df)
    file_scores, subj_scores = zip(*all_metrics)

    if plot_fimp:
        fimp = sorted(zip(test_df.columns[:X_test.shape[1]], clf.feature_importances_), key=lambda l: l[1], reverse=True)
        # [6, 5, 6, 3, 156]
        fl = [0, 6, 11, 17, 20, 176]
        f0_features = [str(x) for x in range(fl[0], fl[1])]
        jitter_features = [str(x) for x in range(fl[1], fl[2])]
        shimmer_features = [str(x) for x in range(fl[2], fl[3])]
        formant_features = [str(x) for x in range(fl[3], fl[4])]
        mfcc_features = [str(x) for x in range(fl[4], fl[5])]

        sum_f0 = sum(val for key, val in fimp if key in f0_features)
        print("Contribution of F0: {:.3f} (avg: {:.3f})".format(sum_f0, sum_f0/len(f0_features)))
        
        sum_jit = sum(val for key, val in fimp if key in jitter_features)
        print("Contribution of Jitter: {:.3f} (avg: {:.3f})".format(sum_jit, sum_jit/len(jitter_features)))
        sum_shim = sum(val for key, val in fimp if key in shimmer_features)
        print("Contribution of Shimmer: {:.3f} (avg: {:.3f})".format(sum_shim, sum_shim/len(shimmer_features)))
        sum_form = sum(val for key, val in fimp if key in formant_features)
        print("Contribution of formants: {:.3f} (avg: {:.3f})".format(sum_form, sum_form/len(formant_features)))
        sum_mfcc = sum(val for key, val in fimp if key in mfcc_features)
        print("Contribution of MFCC: {:.3f} (avg: {:.3f})".format(sum_mfcc, sum_mfcc/len(mfcc_features)))
        print("Total feature importance (should equal to 1):", sum_mfcc + sum_f0 + sum_form + sum_jit + sum_shim)

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


def run_ml_tl_model(scaler, base_X_train, base_X_test, base_y_train, base_y_test, base_df, tgt_df):
    n_features = np.shape(base_X_train)[1]
    base_clf = SGDClassifier()
    base_clf.partial_fit(base_X_train, base_y_train, classes=np.unique(base_y_train))

    base_pos_subjs = list(base_df[base_df['y'] == 1]['subject_id'].unique())
    base_neg_subjs = list(base_df[base_df['y'] == 0]['subject_id'].unique())

    pos_subjs = list(tgt_df[tgt_df['y'] == 1]['subject_id'].unique())
    neg_subjs = list(tgt_df[tgt_df['y'] == 0]['subject_id'].unique())
    max_shot = min(len(pos_subjs), len(neg_subjs)) - 5

    metrics_list, metrics_grouped, n_tgt_train_samples, base_metrics = [], [], [], []
    seed = int(random.random()*10000)
    for n_shots in range(1, max_shot+1):
        scaler_copy = deepcopy(scaler)
        clf = deepcopy(base_clf)  # Copy model trained on base language
        if n_shots > 0:
             # Fine-tune model with pos and neg samples from base and target set
            base_train_df, _ = get_samples(seed, base_pos_subjs, base_neg_subjs, max(1, int(n_shots/6)), base_df)
            tgt_train_df, tgt_test_df = get_samples(seed, pos_subjs, neg_subjs, n_shots, tgt_df)

            # Add target train data to scaler fit
            scaler_copy.partial_fit(tgt_train_df.iloc[:, :n_features].values) 
            tgt_train_df.iloc[:, :n_features] = scaler_copy.transform(tgt_train_df.iloc[:, :n_features].values)
            tgt_test_df.iloc[:, :n_features] = scaler_copy.transform(tgt_test_df.iloc[:, :n_features].values)

            tgt_train_df.reset_index(drop=True, inplace=True) 
            base_train_df.reset_index(drop=True, inplace=True) 
            tgt_train_df = pd.concat([tgt_train_df, base_train_df])

            tgt_X_train = tgt_train_df.iloc[:, :n_features].values
            tgt_y_train = tgt_train_df['y'].values

            clf.partial_fit(tgt_X_train, tgt_y_train)
        else:  # n_shots == 0
            # Use entire tgt set for evaluation
            tgt_test_df = deepcopy(tgt_df)
            tgt_test_df.iloc[:, :n_features] = scaler_copy.transform(tgt_test_df.iloc[:, :n_features].values)

        tgt_X_test = tgt_test_df.iloc[:, :n_features].values
        tgt_y_test = tgt_test_df['y'].values
        base_preds = clf.predict(base_X_test)
        tgt_preds = clf.predict(tgt_X_test)
        tgt_test_df.loc[:, 'preds'] = tgt_preds

        all_metrics = evaluate_predictions(f'RFC ({n_shots} shots)', tgt_y_test, tgt_test_df, base_y_test, base_preds)
        metrics, grouped, base = zip(*all_metrics)
        metrics_list.append(metrics)
        metrics_grouped.append(grouped)
        base_metrics.append(base)
        n_tgt_train_samples.append(n_shots)

        if plot_fimp:
            fimp = sorted(zip(tgt_test_df.columns[:tgt_X_test.shape[1]], np.abs(clf.coef_)), key=lambda l: l[1], reverse=True)
            # [6, 5, 6, 156]
            fl = [0, 6, 11, 17, 20, 176]
            f0_features = [str(x) for x in range(fl[0], fl[1])]
            formant_features = [str(x) for x in range(fl[1], fl[2])]
            jitter_features = [str(x) for x in range(fl[2], fl[3])]
            shimmer_features = [str(x) for x in range(fl[3], fl[4])]
            mfcc_features = [str(x) for x in range(fl[4], fl[5])]

            sum_f0 = np.sum([val for key, val in fimp if key in f0_features])
            print("Contribution of F0: {:.3f} (avg: {:.3f})".format(sum_f0, sum_f0/len(f0_features)))
            sum_form = np.sum([val for key, val in fimp if key in formant_features])
            print("Contribution of formants: {:.3f} (avg: {:.3f})".format(sum_form, sum_form/len(formant_features)))
            sum_jit = np.sum([val for key, val in fimp if key in jitter_features])
            print("Contribution of Jitter: {:.3f} (avg: {:.3f})".format(sum_jit, sum_jit/len(jitter_features)))
            sum_shim = np.sum([val for key, val in fimp if key in shimmer_features])
            print("Contribution of Shimmer: {:.3f} (avg: {:.3f})".format(sum_shim, sum_shim/len(shimmer_features)))
            sum_mfcc = np.sum([val for key, val in fimp if key in mfcc_features])
            print("Contribution of MFCC: {:.3f} (avg: {:.3f})".format(sum_mfcc, sum_mfcc/len(mfcc_features)))
            print("Total feature importance (should equal to 1):", sum_mfcc + sum_f0 + sum_form + sum_jit + sum_shim)

            plt.barh(tgt_test_df.columns[fl[0]:fl[1]], clf.coef_[fl[0]:fl[1]], color='green')
            plt.barh(tgt_test_df.columns[fl[1]:fl[2]], clf.coef_[fl[1]:fl[2]], color='blue')
            plt.barh(tgt_test_df.columns[fl[2]:fl[3]], clf.coef_[fl[2]:fl[3]], color='red')
            plt.barh(tgt_test_df.columns[fl[3]:fl[4]], clf.coef_[fl[3]:fl[4]], color='purple')
            # plt.barh(df.columns[fl[4]:fl[5]], clf.feature_importances_[fl[4]:fl[5]], color='orange')

            plt.barh(tgt_test_df.columns[fl[4]:fl[4]+39], clf.coef_[fl[4]:fl[4]+39], color='green')
            plt.barh(tgt_test_df.columns[fl[4]+39:fl[4]+39+39], clf.coef_[fl[4]+39:fl[4]+39+39], color='blue')
            plt.barh(tgt_test_df.columns[fl[4]+39+39:fl[4]+78+39], clf.coef_[fl[4]+39+39:fl[4]+78+39], color='red')
            plt.barh(tgt_test_df.columns[fl[4]+78+39:fl[4]+78+78], clf.coef_[fl[4]+78+39:fl[4]+78+78], color='orange')

            plt.yticks(
                [(fl[0] + fl[1]) / 2, (fl[1] + fl[2]) / 2, (fl[2] + fl[3]) / 2, (fl[3] + fl[4]) / 2, (fl[4] + fl[5]) / 2],
                ['F0', 'Jitter', 'Shimmer', 'Formants', 'MFCC'])
            plt.xlabel("Relative feature importance")
            plt.tight_layout()
            plt.ylim((fl[0], fl[-1]))
            plt.show()
 
    
    return metrics_list, metrics_grouped, base_metrics, n_tgt_train_samples