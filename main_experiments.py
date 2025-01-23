from run_experiments import run_monolingual, run_crosslingual
from plotting import crosslingual_fstl_comp
import matplotlib.pyplot as plt
import os 

if os.getenv("COLAB_RELEASE_TAG"):  # colab
    experiment_folder = '/content/drive/My Drive/RAIVD_data/experiments/'
else:
    cwd = os.path.abspath(os.getcwd())
    experiment_folder = os.path.join(cwd,'experiments')

def run_all_experiments():
    clf = 'SGD'
    datasets = ['PCGITA', 'NeuroVoz', 'IPVS']
    speech_task = ['tdu', 'ddk', 'sp']	
    ifm_or_nifm = ['vgg', 'hubert0','hubert1']

    for ds in datasets:
        for st in speech_task:
            base_dataset = ds + '_' + st
            for feat in ifm_or_nifm:
                print(f"Started execution of the {clf}-{feat} model with {base_dataset} data ")
                run_monolingual(base_dataset, feat, modeltype=clf, k=5)

# run_all_experiments()
def run_ABCtoD_crosslingual_experiments():
    run_models = False
    models = ['SGD', 'DNN']
    speech_task = ['sp', 'ddk','tdu' ]	
    
    datasets = ['NeuroVoz', 'PCGITA', 'IPVS', 'MDVR']
    ifm_or_nifm = ['ifm', 'hubert0']

    cols = len(datasets)
    rows = len(speech_task)
    fig, ax = plt.subplots(rows, cols, figsize=(cols*4,rows*3))
    subplots = []

    ifm_nifm_clf = ['SGD', 'SGD']  # default values
    for st in speech_task:
        for ds in datasets:
            base = '_'.join([dset for dset in datasets if not dset==ds])

            if 'MDVR' in base and (st == 'sp' or st == 'ddk'):
                base = base.replace('_MDVR', '')
            if ds == 'MDVR' and (st == 'sp' or st == 'ddk'):
                subplots.append(None)
                axis = ax[speech_task.index(st)][datasets.index(ds)]
                axis.set_visible(False)

                continue
            base_dataset = base + '_' + st
            target_dataset = ds + '_' + st
            if run_models:

                ifm_nifm_clf = []
                for feat in ifm_or_nifm:
                    best_clf = None
                    best_performance = 0
                    for clf in models:
                        print(f"Started execution of the {clf}-{feat} model with {base_dataset} and {target_dataset} data {st} task ")

                        performance = run_monolingual(target_dataset, feat, modeltype=clf, k=10)
                        best_clf = clf if best_clf == None or performance > best_performance + 0.03 else best_clf  # slightly favour SGD over DNN due to speed
                        best_performance = performance if performance > best_performance + 0.03 else best_performance
                    ifm_nifm_clf.append(best_clf)

                    
                    run_crosslingual(base_dataset, target_dataset, feat, modeltype=best_clf, k=5)

            print(f"Started plotting {base_dataset} and {target_dataset} data ")
            axis = ax[speech_task.index(st)][datasets.index(ds)]
            print(st, ds, speech_task.index(st), datasets.index(ds), rows, cols)
            if datasets.index(ds) == 0:
                firstcol = True
            else:
                firstcol = False
            if speech_task.index(st) == rows-1:
                lastrow = True
            else:
                lastrow = False
            legend = (speech_task.index(st)==rows-1) and (datasets.index(ds)==cols-1)
            subplot = crosslingual_fstl_comp.plot_TL_performance(base_dataset, target_dataset, ifm_nifm_clf[0],ifm_nifm_clf[1], axis, firstcol, lastrow, legend=legend)
            if legend:
                subplot, legend = subplot
            extent = subplot.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # leg = subplot.get_legend()
            fig.savefig(os.path.join(experiment_folder, f'fstlcomp_{base_dataset}_{target_dataset}'), bbox_inches=extent.expanded(1.2, 1.3), dpi=300)
    fig.add_artist(legend)#, loc='center', bbox_to_anchor=(0,0))
    
    plt.savefig(os.path.join(experiment_folder, 'crosslingual_fstl_comp.png'))
    plt.show()
    
def run_AtoB_crosslingual_experiments():
    run_models = True
    models = ['SGD']#, 'DNN']
    
    datasets = ['NeuroVoz', 'PCGITA', 'IPVS', 'MDVR']
    ifm_or_nifm = ['ifm', 'hubert0']

    cols = len(datasets)
    rows = len(datasets)-1
    fig, ax = plt.subplots(rows, cols, figsize=(cols*4,rows*3))

    ifm_nifm_clf = ['SGD', 'SGD']  # default values
    for ds in datasets:
        for ind, base in enumerate([dset for dset in datasets if not dset==ds]):

            base_dataset = base + '_tdu'
            target_dataset = ds + '_tdu'
            if run_models:

                ifm_nifm_clf = []
                for feat in ifm_or_nifm:
                    best_clf = None
                    best_performance = 0
                    for clf in models:
                        print(f"Started execution of the {clf}-{feat} model with {base_dataset} and {target_dataset} data ")

                        performance = run_monolingual(target_dataset, feat, modeltype=clf, k=10)
                        best_clf = clf if best_clf == None or performance > best_performance + 0.03 else best_clf  # slightly favour SGD over DNN due to speed
                        best_performance = performance if performance > best_performance + 0.03 else best_performance
                    ifm_nifm_clf.append(best_clf)

                    run_crosslingual(base_dataset, target_dataset, feat, modeltype=best_clf, k=5)

            print(f"Started plotting {base_dataset} and {target_dataset} data on {ind} {datasets.index(ds)}")
            axis = ax[ind][datasets.index(ds)]
            # if datasets.index(ds) == 0:
            #     firstcol = True
            # else:
            #     firstcol = False
            # if speech_task.index(st) == rows-1:
            #     lastrow = True
            # else:
            #     lastrow = False
            # legend = (speech_task.index(st)==rows-1) and (datasets.index(ds)==cols-1)
            subplot = crosslingual_fstl_comp.plot_TL_performance(base_dataset, target_dataset, ifm_nifm_clf[0],ifm_nifm_clf[1], axis, True, True, legend=False)
            # if legend:
            #     subplot, legend = subplot
            extent = subplot.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # leg = subplot.get_legend()
            fig.savefig(os.path.join(experiment_folder, f'fstlcomp_{base_dataset}_{target_dataset}'), bbox_inches=extent.expanded(1.2, 1.3), dpi=300)
    # fig.add_artist(legend)#, loc='center', bbox_to_anchor=(0,0))
    
    plt.savefig(os.path.join(experiment_folder, 'crosslingual_fstl_ABcomp.png'))
    plt.show()

run_AtoB_crosslingual_experiments()
# run_ABCtoD_crosslingual_experiments()
