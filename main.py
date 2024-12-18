import configparser

config = configparser.ConfigParser()
config.read('settings.ini')

base_speech_task = config['DATA_SETTINGS']['base_speech_task']
target_speech_task = config['DATA_SETTINGS']['target_speech_task']
base_dataset = config['DATA_SETTINGS']['base_dataset']
target_dataset = config['DATA_SETTINGS']['target_dataset']

ifm_or_nifm = config['EXPERIMENT_SETTINGS']['ifm_or_nifm']
kfolds = config.getint('EXPERIMENT_SETTINGS', 'kfolds')
clf = config['MODEL_SETTINGS']['clf']

recreate_features = config.getboolean('RUN_SETTINGS', 'recreate_features')
run_monolingual = config.getboolean('RUN_SETTINGS', 'run_monolingual')
run_crosslingual = config.getboolean('RUN_SETTINGS', 'run_crosslingual')
run_pretrained = config.getboolean('RUN_SETTINGS', 'run_pretrained')

plot_results = config.getboolean('OUTPUT_SETTINGS', 'plot_results')

base_dataset += '_' + base_speech_task
target_dataset += '_' + target_speech_task

# import run_experiments
# run_experiments.run_experiments()

if recreate_features:
    import get_features
    print(f"Creating {ifm_or_nifm} features for '{base_dataset}' dataset ")
    get_features.create_features(base_dataset, ifm_or_nifm)

    if not target_dataset.startswith('pass') and run_crosslingual:
        print(f"Creating {ifm_or_nifm} features for '{target_dataset}' dataset ")
        get_features.create_features(target_dataset, ifm_or_nifm)    

if run_monolingual:
    import run_experiments
    print(f"Started execution of the {clf}-{ifm_or_nifm} model with {base_dataset} data ")
    run_experiments.run_monolingual(base_dataset, ifm_or_nifm, modeltype=clf, k=kfolds)

if run_crosslingual:
    import run_experiments
    print(f"Started execution of the {clf}-{ifm_or_nifm} model with {base_dataset}-base and {target_dataset}-target data ")
    run_experiments.run_crosslingual(base_dataset, target_dataset, ifm_or_nifm, modeltype=clf, k=kfolds)

if plot_results:
    if clf.endswith('FSTL'):
        from plotting import results_visualised, crosslingual_fstl_comp
        results_visualised.plot_TL_performance(base_dataset, target_dataset)
        crosslingual_fstl_comp.plot_TL_performance(base_dataset, target_dataset, clf)
        
    if run_monolingual:
        #todo: monolingual plotting
        pass

if run_pretrained:
    import pretrained_model
    pretrained_model.run_ptm(base_dataset)