from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns
import configparser

config = configparser.ConfigParser()
config.read('settings.ini')
print_perf_results = config.getboolean('OUTPUT_SETTINGS', 'print_perf_results')
plot_results = config.getboolean('OUTPUT_SETTINGS', 'plot_results')


def evaluate_predictions(clf_name, y_test, preds):
    if print_perf_results:
        print("Prediction scores for the " + clf_name[:-6])
        print("{}-level prediction scores: Accuracy: {:.3f}; AUC: {:.3f}; Sensitivity: {:.3f},; Specificity: {:.3f},"
              .format(clf_name[-6:],
                      accuracy_score(y_test, preds),
                      roc_auc_score(y_test, preds),
                      recall_score(y_test, preds, pos_label=1),
                      recall_score(y_test, preds, pos_label=0)))

        print(confusion_matrix(y_test, preds))
    if plot_results:
        plot_confmat(confusion_matrix(y_test,preds))
    return accuracy_score(y_test, preds), roc_auc_score(y_test, preds), recall_score(y_test, preds, pos_label=1), \
        recall_score(y_test, preds, pos_label=0)


def plot_confmat(conf_mat):
    cm = sns.heatmap(conf_mat, fmt='g', annot=True, cmap=sns.color_palette("light:b", as_cmap=True))
    cm.set(xlabel='Predicted label', ylabel='True Label', title='Confusion Matrix')
    plt.show()
