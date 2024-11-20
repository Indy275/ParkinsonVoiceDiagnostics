from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns
import configparser

config = configparser.ConfigParser()
config.read('settings.ini')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')
print_perf_results = config.getboolean('OUTPUT_SETTINGS', 'print_perf_results')
plot_results = config.getboolean('OUTPUT_SETTINGS', 'plot_results')


def get_scores(clf_name, y_test, preds):
    y_test = list(map(int, y_test))
    preds = list(map(int, preds))
    if print_perf_results:
        print("{} prediction scores: Accuracy: {:.3f}; AUC: {:.3f}; Sensitivity: {:.3f}; Specificity: {:.3f}"
              .format(clf_name,
                      accuracy_score(y_test, preds),
                      roc_auc_score(y_test, preds),
                      recall_score(y_test, preds, pos_label=1),
                      recall_score(y_test, preds, pos_label=0)))

        print(confusion_matrix(y_test, preds))
    # if plot_results:
    #     plot_confmat(confusion_matrix(y_test,preds))
    return float(accuracy_score(y_test, preds)), float(roc_auc_score(y_test, preds)), float(recall_score(y_test, preds, pos_label=1)), \
        float(recall_score(y_test, preds, pos_label=0))


def evaluate_predictions(clf_name, tgt_y, tgt_df, base_y=None, base_df=None):
    metrics = []
    if print_perf_results:
        print(f"Prediction scores for {clf_name}:")
    metrics.append(get_scores('Sample-level', tgt_y.tolist(), tgt_df['preds'].tolist()))
    
    samples_preds = tgt_df.groupby('subject_id').agg({'preds': lambda x: x.mode()[0]}).reset_index()
    samples_ytest = tgt_df.groupby('subject_id').agg({'y': lambda x: x.mode()[0]}).reset_index()

    metrics.append(get_scores('Speaker-level', samples_ytest['y'].tolist(), samples_preds['preds'].tolist()))
    
    if base_y is not None:
        base_preds = base_df.groupby('subject_id').agg({'preds': lambda x: x.mode()[0]}).reset_index()
        base_y = base_df.groupby('subject_id').agg({'y': lambda x: x.mode()[0]}).reset_index()
        metrics.append(get_scores('Base language', base_y['y'].tolist(), base_preds['preds'].tolist()))
    return zip(*metrics)


def plot_confmat(conf_mat):
    cm = sns.heatmap(conf_mat, fmt='g', annot=True, cmap=sns.color_palette("light:b", as_cmap=True))
    cm.set(xlabel='Predicted label', ylabel='True Label', title='Confusion Matrix')
    plt.show()
