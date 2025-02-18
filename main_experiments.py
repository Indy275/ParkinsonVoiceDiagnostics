from run_experiments import run_monolingual, run_crosslingual, run_fs
from plotting import crosslingual_fstl_comp, IFM_NIFM_FSTL_plot
import matplotlib.pyplot as plt
import os 
import pandas as pd

if os.getenv("COLAB_RELEASE_TAG"):  # colab
    experiment_folder = '/content/drive/My Drive/RAIVD_data/experiments/'
else:
    cwd = os.path.abspath(os.getcwd())
    experiment_folder = os.path.join(cwd,'experiments')

def run_mono(models, ds, st, base_dataset, feat):
    best_clf = None
    best_performance = 0
    for clf in models:
        performance = run_monolingual(base_dataset, feat, modeltype=clf, k=10)
        best_clf = clf if best_clf == None or performance > best_performance else best_clf  
        best_performance = performance if performance > best_performance else best_performance
    print(f"Best model for {ds} {feat} is {best_clf} with {best_performance} performance")
    with open(os.path.join(experiment_folder, 'monolingual_result.csv'), 'a') as f:
        result = f'\n{ds},{st},{best_clf},{feat},{best_performance}'
        f.write(result)
    return best_clf

def run_cross(models, ifm_or_nifm, ds, st, base_dataset, target_dataset):
    ifm_nifm_clf = []
    for feat in ifm_or_nifm:
        best_clf = run_mono(models, ds, st, base_dataset, feat)
        ifm_nifm_clf.append(best_clf)
        run_crosslingual(base_dataset, target_dataset, feat, modeltype=best_clf, k=5)


def run_monolingual_experiments():
    models = ['SGD']#, 'DNN']
    datasets = ['PCGITA', 'NeuroVoz', 'IPVS', 'MDVR']
    speech_task = ['tdu']#, 'ddk', 'sp']	
    ifm_or_nifm = ['ifm']#,'hubert0']

    for ds in datasets:
        for st in speech_task:
            base_dataset = ds + '_' + st
            for feat in ifm_or_nifm:
                run_mono(models, ds, st, base_dataset, feat)
                

def get_min_max_improvement(base_dataset,target_dataset, ifm_min, ifm_max, nifm_min, nifm_max, ifm_clf='SGD', nifm_clf='SGD'):
    nifm_model = 'hubert0'
    try:
        metrics_ifm = pd.read_csv(os.path.join(experiment_folder, f'{ifm_clf}_ifm_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
    except:
        ifm_clf = 'DNN' if ifm_clf == 'SGD' else 'SGD'
        metrics_ifm = pd.read_csv(os.path.join(experiment_folder, f'{ifm_clf}_ifm_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
    try:
        metrics_nifm = pd.read_csv(os.path.join(experiment_folder, f'{nifm_clf}_{nifm_model}_metrics_{base_dataset}_{target_dataset}_grouped.csv'))
    except:
        nifm_clf = 'DNN' if nifm_clf == 'SGD' else 'SGD'
        metrics_nifm = pd.read_csv(os.path.join(experiment_folder, f'{nifm_clf}_{nifm_model}_metrics_{base_dataset}_{target_dataset}_grouped.csv'))

    half_data = len(metrics_ifm)//2

    # ifm_imp = metrics_ifm['ROC_AUC'].max() - metrics_ifm['ROC_AUC'].min()
    # nifm_imp = metrics_nifm['ROC_AUC'].max() - metrics_nifm['ROC_AUC'].min()
    ifm_imp = metrics_ifm['ROC_AUC'].iloc[half_data:].max() - metrics_ifm['ROC_AUC'].iloc[half_data:].min()
    nifm_imp = metrics_nifm['ROC_AUC'].iloc[half_data:].max() - metrics_nifm['ROC_AUC'].iloc[half_data:].min()

    # last_sample = len(metrics_ifm)-1
    # ifm_imp = metrics_nifm['ROC_AUC'].iloc[last_sample] - metrics_ifm['ROC_AUC'].iloc[last_sample]

    if ifm_imp < ifm_min:
        ifm_min = ifm_imp
    if ifm_imp > ifm_max:    
        ifm_max = ifm_imp
    if nifm_imp < nifm_min:
        nifm_min = nifm_imp
    if nifm_imp > nifm_max:   
        nifm_max = nifm_imp
    return ifm_min, ifm_max, nifm_min, nifm_max

# run_all_experiments()


def run_selected_crosslingual_experiments():
    ifm_min, nifm_min, ifm_max, nifm_max = 1, 1, 0, 0
    run_models = False
    models = ['SGD']
    speech_task = 'tdu' # ['sp', 'ddk','tdu' ]	
    
    target = 'NeuroVoz'
    base = ['PCGITA',  'IPVS','MDVR']
    ifm_or_nifm = ['ifm']

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    ifm_nifm_clf = ['SGD', 'SGD']  # default values
    base = '_'.join(base)

    base_dataset = base + '_' + speech_task
    target_dataset = target + '_' + speech_task
    if run_models:
        run_fs(target_dataset, 'IFM', modeltype='SGD', k=5)

        # run_cross(models, ifm_or_nifm, target, speech_task, base_dataset, target_dataset)

    print(f"Started plotting {base_dataset} and {target_dataset} data ")
    legend = True
    # subplot = crosslingual_fstl_comp.plot_TL_performance(base_dataset, target_dataset, ifm_nifm_clf[0],ifm_nifm_clf[1], ax, True, True, legend=legend)

    try:
        subplot = crosslingual_fstl_comp.plot_TL_performance(base_dataset, target_dataset, ifm_nifm_clf[0],ifm_nifm_clf[1], ax, True, True, legend=legend)
    except:
        run_cross(models, ifm_or_nifm, target, speech_task, base_dataset, target_dataset)
        subplot = crosslingual_fstl_comp.plot_TL_performance(base_dataset, target_dataset, ifm_nifm_clf[0],ifm_nifm_clf[1], ax, True, True, legend=legend)

    if legend:
        subplot, legend = subplot
        fig.add_artist(legend)
    ifm_min, ifm_max, nifm_min, nifm_max = get_min_max_improvement(base_dataset,target_dataset,  ifm_min, ifm_max, nifm_min, nifm_max)
    print("Min and max improvement:", ifm_min, ifm_max, nifm_min, nifm_max)
    
    plt.savefig(os.path.join(experiment_folder, f'fstlcomp_{base_dataset}_{target_dataset}.pdf'))
    plt.show()


def run_ABCtoD_crosslingual_experiments():
    ifm_min, nifm_min, ifm_max, nifm_max = 1, 1, 0, 0
    run_models = False
    models = ['SGD', 'DNN']
    speech_task = ['sp', 'ddk','tdu' ]	
    
    datasets = ['NeuroVoz', 'PCGITA', 'IPVS', 'MDVR']
    ifm_or_nifm = ['ifm', 'hubert0']

    cols = len(datasets)
    rows = len(speech_task)
    fig, ax = plt.subplots(rows, cols, figsize=(cols*4,rows*3))

    ifm_nifm_clf = ['SGD', 'SGD']  # default values
    for st in speech_task:
        for ds in datasets:
            base = '_'.join([dset for dset in datasets if not dset==ds])
            if 'MDVR' in base and (st == 'sp' or st == 'ddk'):
                base = base.replace('_MDVR', '')
            if ds == 'MDVR' and (st == 'sp' or st == 'ddk'):
                axis = ax[speech_task.index(st)][datasets.index(ds)]
                axis.set_visible(False)
                continue

            base_dataset = base + '_' + st
            target_dataset = ds + '_' + st
            if run_models:
                run_cross(models, ifm_or_nifm, ds, st, base_dataset, target_dataset)

            print(f"Started plotting {base_dataset} and {target_dataset} data ")
            axis = ax[speech_task.index(st)][datasets.index(ds)]
            print(st, ds, speech_task.index(st), datasets.index(ds), rows, cols)
            legend = (speech_task.index(st)==rows-1) and (datasets.index(ds)==cols-1)
            try:    
                subplot = crosslingual_fstl_comp.plot_TL_performance(base_dataset, target_dataset, ifm_nifm_clf[0],ifm_nifm_clf[1], axis, datasets.index(ds) == 0, speech_task.index(st) == rows-1, legend=legend)
            except:
                run_cross(models, ifm_or_nifm, ds, st, base_dataset, target_dataset)
                subplot = crosslingual_fstl_comp.plot_TL_performance(base_dataset, target_dataset, ifm_nifm_clf[0],ifm_nifm_clf[1], axis, datasets.index(ds) == 0, speech_task.index(st) == rows-1, legend=legend)
            
            if legend:
                subplot, legend = subplot
            extent = subplot.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(experiment_folder, f'fstlcomp_{base_dataset}_{target_dataset}'), bbox_inches=extent.expanded(1.2, 1.3), dpi=300)
            ifm_min, ifm_max, nifm_min, nifm_max = get_min_max_improvement(base_dataset,target_dataset,  ifm_min, ifm_max, nifm_min, nifm_max)
    print("Min and max improvement across datasets:", ifm_min, ifm_max, nifm_min, nifm_max)
    fig.add_artist(legend)
    
    plt.savefig(os.path.join(experiment_folder, 'crosslingual_fstl_compifmnifm.pdf'))
    plt.show()
    

def run_both_crosslingual_experiments():
    run_models = False
    models = ['SGD', 'DNN']
    
    datasets = ['NeuroVoz', 'PCGITA', 'IPVS', 'MDVR']
    ifm_or_nifm = ['ifm', 'hubert0']
    st = 'tdu'
    cols = len(datasets)
    rows = len(datasets)
    fig, ax = plt.subplots(rows, cols, figsize=(cols*4,rows*3))

    ifm_nifm_clf = ['SGD', 'SGD']  # default values
    for ds in datasets:
        for ind, base in enumerate([dset for dset in datasets if not dset==ds]):

            base_dataset = base + '_' + st
            target_dataset = ds + '_' + st
            if run_models:
                run_cross(models, ifm_or_nifm, ds, st, base_dataset, target_dataset)

            print(f"Started plotting {base_dataset} and {target_dataset} data on {ind} {datasets.index(ds)}")
            axis = ax[ind][datasets.index(ds)]
            legend = True if ind==0 and datasets.index(ds)==cols-1 else False
            try:
                IFM_NIFM_FSTL_plot.plot_TL_performance(base_dataset, target_dataset, ifm_nifm_clf[0],ifm_nifm_clf[1], axis, datasets.index(ds)==0, False, legend=legend)
            except:
                run_cross(models, ifm_or_nifm, ds, st, base_dataset, target_dataset)
                crosslingual_fstl_comp.plot_TL_performance(base_dataset, target_dataset, ifm_nifm_clf[0],ifm_nifm_clf[1], axis, datasets.index(ds)==0, False, legend=legend)
        axis = ax[rows-1][datasets.index(ds)]
        base_dataset = '_'.join([dset for dset in datasets if not dset==ds]) + '_' + st
        print(f"Started plotting {base_dataset} and {target_dataset} data on {rows-1} {datasets.index(ds)}")
        try:
            IFM_NIFM_FSTL_plot.plot_TL_performance(base_dataset, target_dataset, ifm_nifm_clf[0],ifm_nifm_clf[1], axis, datasets.index(ds)==0, True, legend=False)
        except:
            run_cross(models, ifm_or_nifm, ds, st, base_dataset, target_dataset)
            crosslingual_fstl_comp.plot_TL_performance(base_dataset, target_dataset, ifm_nifm_clf[0],ifm_nifm_clf[1], axis, datasets.index(ds)==0, True, legend=False)    
    plt.savefig(os.path.join(experiment_folder, 'crosslingual_fstl_both_comp.pdf'))
    plt.show()

if __name__ == "__main__":
    # run_monolingual_experiments()

    # run_selected_crosslingual_experiments()

    run_ABCtoD_crosslingual_experiments()

    # run_both_crosslingual_experiments()