from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score


def evaluate_predictions(clf_name, y_test, preds):
    print("Prediction scores for the " + clf_name[:-6])
    print("{}-level pred scores: Accuracy: {:.3f}; AUC: {:.3f}; Sensitivity: {:.3f},; Specificity: {:.3f},".format(
        clf_name[-6:],
        accuracy_score(y_test, preds),
        roc_auc_score(y_test, preds), recall_score(y_test, preds, pos_label=1),
        recall_score(y_test, preds, pos_label=0)))
    print(confusion_matrix(y_test, preds))
