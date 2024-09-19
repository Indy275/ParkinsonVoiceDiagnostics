import sklearn
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score
print("imports done")
X = np.load('C:\\Users\\INDYD\\Documents\\ParkinsonVoiceDiagnostics\\NeuroVoz_preprocessed\\X.npy')
y = np.load('C:\\Users\\INDYD\\Documents\\ParkinsonVoiceDiagnostics\\NeuroVoz_preprocessed\\y.npy')
id = np.load('C:\\Users\\INDYD\\Documents\\ParkinsonVoiceDiagnostics\\NeuroVoz_preprocessed\\id.npy')
train_data = np.load('C:\\Users\\INDYD\\Documents\\ParkinsonVoiceDiagnostics\\NeuroVoz_preprocessed\\train_data.npy')

print("data loading done")
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
X_train = [x for x, is_train in zip(X, train_data) if is_train]
X_test = [x for x, is_train in zip(X, train_data) if not is_train]
y_train = [y for y, is_train in zip(y, train_data) if is_train]
y_test = [y for y, is_train in zip(y, train_data) if not is_train]

print(len(y_train), sum(y_train), len(y_test), sum(y_test))
print("The training data consists of {} samples, of which {} of PD patients ({:.2f}%)".format(len(y_train), sum(y_train),sum(y_train)/len(y_train)*100))
print("The test data consists of {} samples, of which {} of PD patients ({:.2f}%)".format(len(y_test), sum(y_test),sum(y_test)/len(y_test)*100))

rfc = RandomForestClassifier()
for clf in [rfc]:
    clf.fit(X_train,y_train)
    preds = clf.predict(X_test)
    print("Accuracy: ",accuracy_score(y_test, preds))
    print("AUC: ",roc_auc_score(y_test, preds))
    print(confusion_matrix(y_test, preds))