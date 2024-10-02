import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from statistics import mode

from data_util.data_util import load_data, split_data
from eval import evaluate_predictions


def run_ml_model(dataset, ifm_nifm):
    X, y, subj_id, sample_ids, train_data = load_data(dataset, ifm_nifm)
    X_train, X_test, y_train, y_test = split_data(X, y, train_data)
    print(X.shape, y.shape, subj_id.shape, sample_ids.shape, train_data.shape)

    rfc = RandomForestClassifier(random_state=11)
    xgb = XGBClassifier()
    classifiers = [rfc, xgb]
    clf_names = ['Random Forest Classifier', 'XGBoost Classifier']
    for clf, clf_name in zip(classifiers, clf_names):
        print("Fitting {}".format(clf_name))
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        evaluate_predictions(clf_name+'Window', y_test, preds)

        if ifm_nifm[-4] == 'n':  # wiNdow or Nifm; majority voting only sensible for window-level predictions
            print("before sorting")
            print(sum(y_test), len(y_test))
            print(sample_ids)
            print(y_test)
            print(preds)
            sample_ids = list(sample_ids)
            # sorted_lists = sorted(zip(sample_id, y_test, preds), key=lambda i: i[0])
            # sorted_samples, sorted_ytest, sorted_preds = zip(*sorted_lists)
            # sorted_samples = list(sorted_samples)
            # sorted_ytest = list(sorted_ytest)
            # sorted_preds = list(sorted_preds)

            samples_ytest, samples_preds = [], []
            # loop through unique items
            for sampleid in np.unique(sample_ids):
                i = sample_ids.index(sampleid)  # Get first occurrence of sample_id
                x = sample_ids.count(sampleid)  # Get length of sample
                sample_prediction = mode(preds[i:i+x])
                print("sampleid:", sampleid)
                print(y_test[i:i+x])
                print(preds[i:i+x])
                samples_ytest.append(y_test[i])
                samples_preds.append(sample_prediction)
        if clf_name == 'Random Forest Classifier':

            print(clf.feature_importances_)
        break

            # evaluate_predictions(clf_name+'Sample', samples_ytest, samples_preds)
