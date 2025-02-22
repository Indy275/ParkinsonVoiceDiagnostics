import os
import random
from copy import deepcopy
import numpy as np
import pandas as pd
import configparser
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from data_util import load_data, scale_features, get_samples
from eval import evaluate_predictions

from DNN_models import DNN_model, CNN_model, PT_model, ResNet_model
from ML_models import SGD_model

config = configparser.ConfigParser()
config.read('settings.ini')
plot_fimp = config.getboolean('OUTPUT_SETTINGS', 'plot_fimp')
print_intermediate = config.getboolean('OUTPUT_SETTINGS', 'print_intermediate')
clf = config['MODEL_SETTINGS']['clf']
fewshot = config.getboolean('EXPERIMENT_SETTINGS', 'fewshot')

gender = config.getint('EXPERIMENT_SETTINGS', 'gender')
tgt_gender = config.getint('EXPERIMENT_SETTINGS', 'tgt_gender')

pd.options.mode.chained_assignment = None  # default='warn'

if os.getenv("COLAB_RELEASE_TAG"):  # colab
    experiment_folder = '/content/drive/My Drive/RAIVD_data/experiments/'
else:
    cwd = os.path.abspath(os.getcwd())
    experiment_folder = os.path.join(cwd,'experiments')
if not os.path.exists(experiment_folder):
    os.makedirs(experiment_folder)
    df = pd.DataFrame(columns=['dataset','model','ifm_nifm','fMACC','fMAUC','sMACC','sMAUC'])
    df.to_csv(os.path.join(experiment_folder,'mono_result.csv'), index=False)

    df.to_csv(os.path.join(experiment_folder,'crosslingual_result.csv'), index=False)


def get_model(modeltype):
    model_dict = {
        'SGD': SGD_model,
        'DNN': DNN_model,
        'CNN': CNN_model,
        'PTM': PT_model,
        'ResNet': ResNet_model
        }
    modelclass = model_dict[modeltype]
    model = modelclass(mono=True)
    return model


def prepare_data(dataset, ifm_nifm, gender, addition=''):
    dfs, n_feat = [], []
    for ds in dataset.split('_')[:-1]:
        df, n_features = load_data(f"{ds}_{dataset.split('_')[-1]}", ifm_nifm)
        n_feat.append(n_features)
        df['subject_id'] = df['subject_id'].astype(str) + df['dataset']
        dfs.append(df)
    df = pd.concat(dfs)
    df['sample_id'] = df['sample_id'].astype(str) + addition
    df.reset_index(drop=True, inplace=True)
    assert len(set(n_feat)) == 1, "Number of features across datasets should be equal: {}".format(n_feat)

    # Experiment: only include Male/Female participants
    if gender < 2:
        df = df[df['gender']==gender]

    return df, n_features


def prepare_train_test(base_model, base_df, base_features, k):
    base_df_split = base_df.drop_duplicates(['subject_id'])
    base_df_split.loc[:,'ygender'] = base_df_split['y'].astype(str) + '_' + base_df_split['gender'].astype(str) #+ '_' + base_df_split['dataset'].astype(str)
    k = min(k, np.min(list(np.unique(base_df_split['ygender'], return_counts=True)[1])))
    
    data_splits = []
    outer_k = 10
    for _ in range(outer_k):
        kf = StratifiedKFold(n_splits=k, shuffle=True)
        for train_split_indices, test_split_indices in kf.split(base_df_split['subject_id'], base_df_split['ygender']):
            base_df_copy = deepcopy(base_df)
            model = deepcopy(base_model)

            train_subjects = base_df_split.iloc[train_split_indices]['subject_id']
            test_subjects = base_df_split.iloc[test_split_indices]['subject_id']
            train_indices = base_df_copy[base_df_copy['subject_id'].isin(train_subjects)].index.tolist()
            test_indices = base_df_copy[base_df_copy['subject_id'].isin(test_subjects)].index.tolist()
            scaler, base_df_copy = scale_features(base_df_copy, base_features, train_indices, test_indices)
            data_splits.append([scaler, model, base_df_copy, train_indices, test_indices])
    return data_splits


def get_fimp(model, data):
    if not model.startswith("SGD"):
        X, y = data.dataset[:]
    else:
        X, y = data
    clf = RandomForestClassifier()
    clf.fit(X, y)
    fimp = clf.feature_importances_
    return fimp


def run_monolingual(dataset, ifm_nifm, modeltype, k=2):
    model = get_model(modeltype)
    
    df, n_features = prepare_data(dataset, ifm_nifm, gender)
    
    if print_intermediate:
        print(f"Data loaded succesfully with shapes {df.shape}, now running {model.name} classifier with {k} folds")

    data_splits = prepare_train_test(model, df, n_features, k)

    file_metrics, subject_metrics, fimps = [], [], []
    for split in tqdm(data_splits):
        scaler, model, base_df, base_train_idc, base_test_idc = split
        # Create base train and test set based on split indices
        train_df = base_df.loc[base_train_idc, :]
        test_df = base_df.loc[base_test_idc, :]

        # Experiment with multi-lingual dataset
        if False: 
            dataset = 'IPVS'  # only test on specific data
            # train_df = train_df[train_df['dataset']!=dataset]
            test_df = test_df[test_df['dataset']==dataset]

        if print_intermediate:
            train_df = (base_df.loc[base_train_idc, :])
            test_df = (base_df.loc[base_test_idc, :])
            for dataset_name, group in train_df.groupby('dataset'):
                print(f"Train Dataset: {dataset_name}")
                print("Subject IDs:", np.sort(group['subject_id'].unique()))
                print()
            for dataset_name, group in test_df.groupby('dataset'):
                print(f"Test Dataset: {dataset_name}")
                print("Subject IDs:", np.sort(group['subject_id'].unique()))
                print()
            print("Train %PD:",round(base_df.loc[base_train_idc, 'y'].sum()/ len(base_train_idc),3))
            print("Test %PD:",round(base_df.loc[base_test_idc, 'y'].sum()/ len(base_test_idc),3))
            print("Train %male",round(base_df.loc[base_train_idc, 'gender'].sum()/ len(base_train_idc),3))
            print("Test %male",round(base_df.loc[base_test_idc, 'gender'].sum()/ len(base_test_idc),3)) 
        
        # Get base train data
        base_train_df, train_loader, n_features = model.get_X_y(train_df, train=True)

        # Create model instance
        model.create_model(n_features)

        # Train model with base train data
        model.train(train_loader)

        # Prepare data for evaluation
        test_df, X_test, y_test = model.get_X_y(test_df)

        # Evaluate model
        preds = model.eval_monolingual(X_test)
        test_df.loc[:, 'preds'] = preds
        all_metrics = evaluate_predictions(f'{model.name}', y_test, test_df)
        if plot_fimp:
            fimp = get_fimp(modeltype, train_loader)
            fimps.append(fimp)
        
        metrics, grouped = zip(*all_metrics)
        file_metrics.append(metrics)
        subject_metrics.append(grouped)


    file_scores, subject_scores = [], []
    score_names = ['Mean Acc:', 'Mean AUC:', 'Mean Sens:', 'Mean Spec:']
    for metric in np.mean(file_metrics, axis=0).flatten():
        file_scores.append(round(metric, 3))
    for metric in np.mean(subject_metrics, axis=0).flatten():
        subject_scores.append(round(metric, 3))

    if print_intermediate:
        print(f"Average {k}-fold performance of {model.name}-{ifm_nifm} model with {dataset} data:")
        if dataset[-3:] != 'tdu' and dataset[-3:] != 'ddk':
            print("Target data (file-level) performance:")
            for name, score in zip(score_names, file_scores):
                print(name, score)

        print("Target data (speaker-level) performance:")
        for name, score in zip(score_names, subject_scores):
                print(name, score)
    print(f"Mean {k}-fold AUC {model.name}-{ifm_nifm} on {dataset}:", subject_scores[1])
    
    if k >= 5:   # write results only when at least 5-fold crossvalidation is done
        with open(os.path.join(experiment_folder, 'mono_result.csv'), 'a') as f:
            result = f'\n{dataset},{model.name},{ifm_nifm},{file_scores[0]},{file_scores[1]},{subject_scores[0]},{subject_scores[1]}'
            f.write(result)

    if plot_fimp:
        from plotting.results_visualised import fimp_plot, fimp_plot_nifm
        fimp = np.mean(fimps, axis=0)
        fimp_plot(fimp, df)
        # fimp_plot_nifm(fimp, df)

    return subject_scores[1]



def run_fewshot(scaler, model, base_train_df, base_test_df, tgt_df, n_features):
    base_df = pd.concat([base_train_df, base_test_df], ignore_index=True)
    base_pos_subjs = list(base_df[base_df['y'] == 1]['subject_id'].unique())
    base_neg_subjs = list(base_df[base_df['y'] == 0]['subject_id'].unique())

    pos_subjs = list(tgt_df[tgt_df['y'] == 1]['subject_id'].unique())
    neg_subjs = list(tgt_df[tgt_df['y'] == 0]['subject_id'].unique())
    if fewshot:
        min_shot = 0
        max_shot = min(len(pos_subjs), len(neg_subjs)) -1  # 1 less than the number of subjects
    else:  
        tgt_train_size = int(min(len(pos_subjs), len(neg_subjs))*0.7) # 70% of the data
        min_shot = tgt_train_size
        max_shot = tgt_train_size + 1 
    
    metrics_list, metrics_grouped, n_tgt_train_samples, base_metrics = [], [], [], []
    seed = int(random.random()*10000)
    for n_shots in range(min_shot, max_shot):
        base_test_df_copy = deepcopy(base_test_df)
        tgt_test_df = deepcopy(tgt_df)
        scaler_copy = deepcopy(scaler)
        model_copy = deepcopy(model)

        if n_shots > 0:
            # Fine-tune model with pos and neg samples from base and target set
            base_mixing_parameter = 0.5
            n_base_samples = int(max(1, min(base_mixing_parameter*n_shots, len(base_pos_subjs) -1, len(base_neg_subjs) -1)))
            base_train_df, base_test_df_copy = get_samples(seed, base_pos_subjs, base_neg_subjs, n_base_samples, base_df)
            tgt_train_df, tgt_test_df = get_samples(seed, pos_subjs, neg_subjs, n_shots, tgt_df)

            # Add target train data to scaler fit
            scaler_copy.partial_fit(tgt_train_df.iloc[:, :n_features].values) 
            tgt_train_df.iloc[:, :n_features] = scaler_copy.transform(tgt_train_df.iloc[:, :n_features].values)

            # Concatenate train data
            # tgt_train_df = pd.concat([tgt_train_df, base_train_df], ignore_index=True, axis=0)
            # Get target train data
            tgt_train_df, train_loader, n_features = model_copy.get_X_y(tgt_train_df, train=True)
            
            # Fine-tune model using target data
            model_copy.train(train_loader)

        if n_shots == 0:
            continue
        
        # Prepare data for evaluation
        tgt_test_df.iloc[:, :n_features] = scaler_copy.transform(tgt_test_df.iloc[:, :n_features].values)
        tgt_test_df, tgt_X_test, tgt_y_test = model_copy.get_X_y(tgt_test_df)
        base_test_df_copy, base_X_test, base_y_test = model_copy.get_X_y(base_test_df_copy)
        
        # Evaluate model
        base_preds, tgt_preds = model_copy.eval_multilingual(base_X_test, tgt_X_test)
        base_test_df_copy.loc[:, 'preds'] = base_preds
        tgt_test_df.loc[:, 'preds'] = tgt_preds

        all_metrics = evaluate_predictions(f'{model.name} ({n_shots} shots)', tgt_y_test, tgt_test_df, base_y_test, base_test_df_copy)
        
        metrics, grouped, base = zip(*all_metrics)
        metrics_list.append(metrics)
        metrics_grouped.append(grouped)
        base_metrics.append(base)
        n_tgt_train_samples.append(n_shots)

    return zip(*[metrics_list, metrics_grouped, base_metrics, n_tgt_train_samples])


def run_crosslingual(base_dataset, target_dataset, ifm_nifm, modeltype, k=2):
    model = get_model(modeltype)

    base_df, n_features = prepare_data(base_dataset, ifm_nifm, gender, addition='base')
    target_df, tgt_features = prepare_data(target_dataset, ifm_nifm, tgt_gender, addition='tgt')
    assert n_features == tgt_features, "Number of features across languages should be equal: {} and {}".format(
        n_features, tgt_features)
    if print_intermediate:
        print(f"Data loaded succesfully with shapes {base_df.shape}, {target_df.shape}, now running {modeltype} classifier")
    
    data_splits = prepare_train_test(model, base_df, n_features, k)

    file_metrics, subject_metrics, base_metrics = [], [], []
    for split in tqdm(data_splits):
    
        scaler, model, base_df, base_train_idc, base_test_idc = split
        
        # Create base train and test set based on split indices
        base_train_df = base_df.loc[base_train_idc, :]
        base_test_df = base_df.loc[base_test_idc, :]

        if print_intermediate:
            print("Train subjects:", np.sort(base_df.loc[base_train_idc, 'subject_id'].unique()))
            print("Test subjects:", np.sort(base_df.loc[base_test_idc, 'subject_id'].unique()))
            print("Train %PD:",round(base_df.loc[base_train_idc, 'y'].sum()/ len(base_train_idc),3))
            print("Test %PD:",round(base_df.loc[base_test_idc, 'y'].sum()/ len(base_test_idc),3))
            print("Train %male",round(base_df.loc[base_train_idc, 'gender'].sum()/ len(base_train_idc),3))
            print("Test %male",round(base_df.loc[base_test_idc, 'gender'].sum()/ len(base_test_idc),3)) 
        
        # Get base train data
        base_train_df, train_loader, n_features = model.get_X_y(base_train_df, train=True)

        # Create model instance
        model.create_model(n_features)

        # Train model with base train data
        model.train(train_loader)

        metrics = run_fewshot(scaler, model, base_train_df, base_test_df, target_df, n_features)
        file_metric, subject_metric, base_metric, n_tgt_train_samples = zip(*metrics)

        if print_intermediate:
            print(f"Average result:\nFile metrics:",np.mean(file_metric, axis=0))
            print("Subject metrics:",np.mean(subject_metric, axis=0),"\nBase metrics:",np.mean(base_metric, axis=0))

        file_metrics.append(file_metric)
        subject_metrics.append(subject_metric)
        base_metrics.append(base_metric)

    if fewshot:
        fmetrics_df = pd.DataFrame(np.mean(file_metrics, axis=0), columns=['Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity'])
        fmetrics_df['Iteration'] = n_tgt_train_samples
        fmetrics_df.to_csv(os.path.join(experiment_folder, f'{modeltype}_{ifm_nifm}_metrics_{base_dataset}_{target_dataset}.csv'), index=False)

        smetrics_df = pd.DataFrame(np.mean(subject_metrics,axis=0), columns=['Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity'])
        smetrics_df['Iteration'] = n_tgt_train_samples
        smetrics_df.to_csv(os.path.join(experiment_folder, f'{modeltype}_{ifm_nifm}_metrics_{base_dataset}_{target_dataset}_grouped.csv'), index=False)
        
        base_metrics_df = pd.DataFrame(np.mean(base_metrics,axis=0), columns=['Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity'])
        base_metrics_df['Iteration'] = n_tgt_train_samples
        base_metrics_df.to_csv(os.path.join(experiment_folder, f'{modeltype}_{ifm_nifm}_metrics_{base_dataset}_{target_dataset}_base.csv'), index=False)
        
        print(f'Metrics saved to: {experiment_folder}{modeltype}_{ifm_nifm}_metrics_{base_dataset}_{target_dataset}.csv')
    else:  # No Few-Shot
        file_scores, subject_scores, base_scores = [], [], []
        score_names = ['Mean Acc:', 'Mean AUC:', 'Mean Sens:', 'Mean Spec:']
        for metric in np.mean(file_metrics, axis=0).flatten():
            file_scores.append(round(metric, 3))
        for metric in np.mean(subject_metrics, axis=0).flatten():
            subject_scores.append(round(metric, 3))
        for metric in np.mean(base_metrics, axis=0).flatten():
            base_scores.append(round(metric, 3))

        if base_dataset[-3:] != 'tdu' and base_dataset[-3:] != 'ddk':
            print("Target data (file-level) performance:")
            for name, score in zip(score_names, file_scores):
                print(name, score)

        print("Target data (speaker-level) performance:")
        for name, score in zip(score_names, subject_scores):
                print(name, score)
        
        print("Base data performance:")
        for name, score in zip(score_names, base_scores):
                print(name, score)
        
        if k >= 5:   # write results of at least 5-fold crossvalidated results
            with open(os.path.join(experiment_folder,'crosslingual_result.csv'), 'a') as f:
                result = f'\n{base_dataset}_{target_dataset},{model.name},{ifm_nifm},{file_scores[0]},{file_scores[1]},{subject_scores[0]},{subject_scores[1]}'
                f.write(result)

def run_fs(target_dataset, ifm_nifm, modeltype, k=2):
    model = get_model(modeltype)

    target_df, tgt_features = prepare_data(target_dataset, ifm_nifm, tgt_gender, addition='tgt')
    data_splits = prepare_train_test(model, target_df, tgt_features, k)
    file_metrics, subject_metrics, base_metrics = [], [], []
    for split in tqdm(data_splits):
    
        scaler, model, base_df, base_train_idc, base_test_idc = split
        model.create_model(tgt_features)
        scaler = StandardScaler()
        metrics = run_fewshot(scaler, model, target_df, target_df, target_df, tgt_features)
        file_metric, subject_metric, base_metric, n_tgt_train_samples = zip(*metrics)

        if print_intermediate:
            print(f"Average result:\nFile metrics:",np.mean(file_metric, axis=0))
            print("Subject metrics:",np.mean(subject_metric, axis=0),"\nBase metrics:",np.mean(base_metric, axis=0))

        file_metrics.append(file_metric)
        subject_metrics.append(subject_metric)
        base_metrics.append(base_metric)

    if fewshot:
        fmetrics_df = pd.DataFrame(np.mean(file_metrics, axis=0), columns=['Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity'])
        fmetrics_df['Iteration'] = n_tgt_train_samples
        fmetrics_df.to_csv(os.path.join(experiment_folder, f'{modeltype}_{ifm_nifm}_metrics_fewshot_{target_dataset}.csv'), index=False)

        smetrics_df = pd.DataFrame(np.mean(subject_metrics,axis=0), columns=['Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity'])
        smetrics_df['Iteration'] = n_tgt_train_samples
        smetrics_df.to_csv(os.path.join(experiment_folder, f'{modeltype}_{ifm_nifm}_metrics_fewshot_{target_dataset}_grouped.csv'), index=False)
        
        base_metrics_df = pd.DataFrame(np.mean(base_metrics,axis=0), columns=['Accuracy', 'ROC_AUC', 'Sensitivity', 'Specificity'])
        base_metrics_df['Iteration'] = n_tgt_train_samples
        base_metrics_df.to_csv(os.path.join(experiment_folder, f'{modeltype}_{ifm_nifm}_metrics_fewshot_{target_dataset}_base.csv'), index=False)
        
        print(f'Metrics saved to: {experiment_folder}{modeltype}_{ifm_nifm}_metrics_fewshot_{target_dataset}.csv')
    else:  # No Few-Shot
        file_scores, subject_scores, base_scores = [], [], []
        score_names = ['Mean Acc:', 'Mean AUC:', 'Mean Sens:', 'Mean Spec:']
        for metric in np.mean(file_metrics, axis=0).flatten():
            file_scores.append(round(metric, 3))
        for metric in np.mean(subject_metrics, axis=0).flatten():
            subject_scores.append(round(metric, 3))
        for metric in np.mean(base_metrics, axis=0).flatten():
            base_scores.append(round(metric, 3))

        print("Target data (speaker-level) performance:")
        for name, score in zip(score_names, subject_scores):
                print(name, score)
        
        print("Base data performance:")
        for name, score in zip(score_names, base_scores):
                print(name, score)
        
        if k >= 5:   # write results of at least 5-fold crossvalidated results
            with open(os.path.join(experiment_folder,'crosslingual_result.csv'), 'a') as f:
                result = f'\nfewshot_{target_dataset},{model.name},{ifm_nifm},{file_scores[0]},{file_scores[1]},{subject_scores[0]},{subject_scores[1]}'
                f.write(result)



def run_dataset_separation():
    # Experiment: train model to separate data sets
    datasets = ['NeuroVoz', 'PCGITA', 'IPVS']
    dfs = []
    for i, ds in enumerate(datasets):
        print(ds)
        ds += '_sp'
        df, n_features = load_data(ds, 'ifm')
        df = df[df['y']==0]
        df.loc[:, 'y'] = i
        dfs.append(df)

    df = pd.concat(dfs)

    model = RandomForestClassifier()
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    base_df = df.drop_duplicates(['subject_id'])

    accs, rocs = [], []
    conf_mats = []
    for train_split_indices, test_split_indices in kf.split(base_df['subject_id'], base_df['dataset']):
        train_subjects = base_df.iloc[train_split_indices]['subject_id']
        test_subjects = base_df.iloc[test_split_indices]['subject_id']
        train_indices = df[df['subject_id'].isin(train_subjects)].index.tolist()
        test_indices = df[df['subject_id'].isin(test_subjects)].index.tolist()
        X_train = df.iloc[train_indices, :n_features].values
        y_train = df.iloc[train_indices, :]['y'].values
        X_test = df.iloc[test_indices, :n_features].values
        y_test = df.iloc[test_indices, :]['y'].values

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)
        preds2 = model.predict(X_test)
        acc = accuracy_score(y_test, preds2)
        conf_mat = confusion_matrix(y_test, preds2)
        roc_auc = roc_auc_score(y_test, preds, multi_class='ovo')
        accs.append(acc)
        rocs.append(roc_auc)
        conf_mats.append(conf_mat)
    
    print("Accuracy:", np.mean(accs))
    print("ROC AUC:", np.mean(rocs))
    print(np.shape(conf_mats))
    print(conf_mats)
    print(np.mean(conf_mats, axis=0))
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=datasets,
        cmap=plt.cm.Blues,
        normalize='true',
        ax=ax
    )
    fig.savefig(os.path.join(experiment_folder, 'dataset_separation_hubert.pdf'))
    plt.show()