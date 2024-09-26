import numpy as np
from statistics import mode
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

dataset = 'Italian2'  # NeuroVoz Czech Italian test


def load_data(dataset):
    store_location = 'C:\\Users\\INDYD\\Documents\\ParkinsonVoiceDiagnostics\\{}_preprocessed\\'.format(dataset)

    X = np.load(store_location + 'X.npy')
    y = np.load(store_location + 'y.npy')
    subj_id = np.load(store_location + 'subj_id.npy')
    sample_id = np.load(store_location + 'sample_id.npy')
    train_data = np.load(store_location + 'train_data.npy')
    return X, y, subj_id, sample_id, train_data


X, y, subj_id, sample_id, train_data = load_data(dataset)

X_train = [x for x, is_train in zip(X, train_data) if is_train]
X_test = [x for x, is_train in zip(X, train_data) if not is_train]
y_train = [y for y, is_train in zip(y, train_data) if is_train]
y_test = [y for y, is_train in zip(y, train_data) if not is_train]

test_id = [y for y, is_train in zip(sample_id, train_data) if not is_train]

print(
    "The training data consists of {} samples, of which {} of PD patients ({:.1f}%)".format(len(y_train), sum(y_train),
                                                                                            sum(y_train) / len(
                                                                                                y_train) * 100))
print("The test data consists of {} samples, of which {} of PD patients ({:.1f}%)".format(len(y_test), sum(y_test),
                                                                                          sum(y_test) / len(
                                                                                              y_test) * 100))

rfc = RandomForestClassifier()
xgb = XGBClassifier()
classifiers = [rfc, xgb]
clf_names = ['Random Forest Classifier', 'XGBoost Classifier']
for clf, clf_name in zip(classifiers,clf_names) :
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Prediction scores for the "+clf_name)
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

    print("File-level prediction scores: Accuracy: {:.3f}; AUC: {:.3f}".format(accuracy_score(sorted_ytest, sorted_preds),
                                                                             roc_auc_score(sorted_ytest, sorted_preds)))
    print(confusion_matrix(sorted_ytest, sorted_preds))
