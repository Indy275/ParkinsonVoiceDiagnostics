from run_experiments import run_monolingual, run_crosslingual
from plotting import crosslingual_fstl_comp
import matplotlib.pyplot as plt

def run_all_experiments():
    clf = 'SGD'
    datasets = ['PCGITA', 'NeuroVoz', 'IPVS']
    speech_task = ['tdu', 'ddk', 'a']	
    ifm_or_nifm = ['vgg', 'hubert0','hubert1']

    for ds in datasets:
        for st in speech_task:
            base_dataset = ds + '_' + st
            for feat in ifm_or_nifm:
                print(f"Started execution of the {clf}-{feat} model with {base_dataset} data ")
                run_monolingual(base_dataset, feat, modeltype=clf, k=5)

run_all_experiments()
def run_all_crosslingual_experiments():
    models = ['SGD', 'SGD'] # change to 'DNN'
    speech_task = ['a','ddk','tdu']	
    base_ds = ['PCGITA_IPVS_MDVR','NeuroVoz_IPVS_MDVR','NeuroVoz_PCGITA_MDVR','NeuroVoz_PCGITA_IPVS'] #'NeuroVozPCGITAIPVS'
    
    datasets = ['NeuroVoz', 'PCGITA', 'IPVS', 'MDVR']
    ifm_or_nifm = ['ifm', 'vgg']

    cols = len(datasets)
    rows = len(speech_task)
    fig, ax = plt.subplots(rows, cols, figsize=(12, 10))
    subplots = []

    for st in speech_task:
        for base, ds in zip(base_ds, datasets):
            # print(base, ds, st)
            if 'MDVR' in base and (st == 'a' or st == 'ddk'):
                base = base.replace('_MDVR', '')
            if ds == 'MDVR' and (st == 'a' or st == 'ddk'):
                subplots.append(None)
                # axis = ax[speech_task.index(st)][datasets.index(ds)]
                crosslingual_fstl_comp.plot_TL_performance(base + '_tdu', ds + '_tdu', 'SGD', ax[speech_task.index(st)][datasets.index(ds)], firstcol=False, lastrow=False, legend=True)

                continue
            base_dataset = base + '_' + st

            target_dataset = ds + '_' + st
            for feat in ifm_or_nifm:
                best_clf = None
                best_performance = 0
                for clf in models:
                    print(f"Started execution of the {clf}-{feat} model with {base_dataset} and {target_dataset} data {st} task ")

                    performance = run_monolingual(target_dataset, feat, modeltype=clf, k=5)
                    best_clf = clf if best_clf == None or performance > best_performance else best_clf
                    best_performance = performance if performance > best_performance else best_performance
                
                run_crosslingual(base_dataset, target_dataset, feat, modeltype=best_clf, k=2)

            print(f"Started plotting {base_dataset} and {target_dataset} data {st} task ")
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
            subplot = crosslingual_fstl_comp.plot_TL_performance(base_dataset, target_dataset, 'SGD', axis, firstcol, lastrow, legend=False)
            subplots.append(subplot)
    plt.show()
    

run_all_crosslingual_experiments()
