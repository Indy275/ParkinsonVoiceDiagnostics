from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
import pandas as pd
from data_util.data_util import load_data
from eval import evaluate_predictions
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'


def run_ml_model(dataset, ifm_nifm):
    df, n_features = load_data(dataset, ifm_nifm)
    df['train_test'] = df['train_test'].astype(bool)
    X_train = df[df['train_test']].iloc[:, :n_features]
    X_test = df[~df['train_test']].iloc[:, :n_features]
    y_train = df[df['train_test']]['y']
    y_test = df[~df['train_test']]['y']

    print("Train subjects:", np.sort(df[df['train_test']]['subject_id'].unique()))
    print("Test subjects:", np.sort(df[~df['train_test']]['subject_id'].unique()))

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    rfc = RandomForestClassifier(random_state=1)
    # xgb = XGBClassifier()
    classifiers = [rfc]  # , xgb]
    clf_names = ['Random Forest Classifier']  # , 'XGBoost Classifier']
    for clf, clf_name in zip(classifiers, clf_names):
        print("Fitting {}".format(clf_name))
        clf.fit(X_train, y_train)
        test_df = df[~df['train_test']]
        preds = clf.predict(X_test)
        test_df.loc[:, 'preds'] = preds

        evaluate_predictions(clf_name + 'Window', y_test.tolist(), test_df['preds'].tolist())

        # if ifm_nifm[-4] == 'n':  # wiNdow or Nifm; majority voting only sensible for window-level predictions
        samples_preds = test_df.groupby('subject_id').agg({'preds': lambda x: x.mode()[0]}).reset_index()
        samples_ytest = test_df.groupby('subject_id').agg({'y': lambda x: x.mode()[0]}).reset_index()

        evaluate_predictions(clf_name + 'SubjID', samples_ytest['y'].tolist(), samples_preds['preds'].tolist())

        if clf_name == 'Random Forest Classifier':
            print(sorted(zip(df.columns[:n_features], clf.feature_importances_), key=lambda l: l[1], reverse=True))
            fimp = sorted(zip(df.columns[:n_features], clf.feature_importances_), key=lambda l: l[1], reverse=True)

            # [156, 6, 3, 5, 6]
            mfcc_features = [str(x) for x in range(0, 156)]
            f0_features = [str(x) for x in range(156, 162)]
            formant_features = [str(x) for x in range(162, 165)]
            jitter_features = [str(x) for x in range(165, 169)]
            shimmer_features = [str(x) for x in range(169, 176)]

            sum_mfcc = sum(val for key, val in fimp if key in mfcc_features)
            print("Contribution of MFCC:", sum_mfcc)
            sum_f0 = sum(val for key, val in fimp if key in f0_features)
            print("Contribution of F0:", sum_f0)
            sum_form = sum(val for key, val in fimp if key in formant_features)
            print("Contribution of formants:", sum_form)
            sum_jit = sum(val for key, val in fimp if key in jitter_features)
            print("Contribution of Jitter:", sum_jit)
            sum_shim = sum(val for key, val in fimp if key in shimmer_features)
            print("Contribution of Shimmer:", sum_shim)
            print("Total feature importance (should equal to 1):", sum_mfcc + sum_f0 + sum_form + sum_jit + sum_shim)
        break  # Remove this line
