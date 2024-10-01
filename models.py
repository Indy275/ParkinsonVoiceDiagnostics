import numpy as np
from statistics import mode
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from data_util.data_util import load_data, split_data


def run_ml_model(dataset, ifm_nifm):
    X, y, subj_id, sample_id, train_data = load_data(dataset, ifm_nifm)
    X_train, X_test, y_train, y_test = split_data(X, y, train_data)

    rfc = RandomForestClassifier()
    xgb = XGBClassifier()
    classifiers = [rfc, xgb]
    clf_names = ['Random Forest Classifier', 'XGBoost Classifier']
    for clf, clf_name in zip(classifiers, clf_names):
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        print("Prediction scores for the " + clf_name)
        print("Sample-level prediction scores: Accuracy: {:.3f}; AUC: {:.3f}".format(accuracy_score(y_test, preds),
                                                                                     roc_auc_score(y_test, preds)))
        print(confusion_matrix(y_test, preds))

        sorted_lists = sorted(zip(sample_id, preds, y_test))
        sorted_samples, sorted_preds, sorted_ytest = zip(*sorted_lists)
        sorted_samples = list(sorted_samples)
        sorted_preds = list(sorted_preds)
        sorted_ytest = list(sorted_ytest)

        for i in sorted_samples:
            sample = sorted_samples[i]
            x = sorted_samples.count(sample)
            pred = sorted_preds[i:i + x]
            most_common_val = mode(pred)
            sorted_preds[i:i + x] = [most_common_val] * x

        print("File-level prediction scores: Accuracy: {:.3f}; AUC: {:.3f}".format(
            accuracy_score(sorted_ytest, sorted_preds),
            roc_auc_score(sorted_ytest, sorted_preds)))
        print(confusion_matrix(sorted_ytest, sorted_preds))
