import configparser

config = configparser.ConfigParser()
config.read('settings.ini')

speech_task = config['DATA_SETTINGS']['speech_task']
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

if speech_task == 'tdu':
    base_dataset += 'tdu'
    target_dataset += 'tdu'
elif speech_task == 'ddk':
    base_dataset += 'ddk'
    target_dataset += 'ddk'
elif speech_task == 'lr':
    base_dataset += 'lr'
    target_dataset += 'lr'

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
    run_experiments.run_monolingual(base_dataset, ifm_or_nifm, model=clf, k=kfolds)

if run_crosslingual:
    import run_experiments
    print(f"Started execution of the {clf}-{ifm_or_nifm} model with {base_dataset}-base and {target_dataset}-target data ")
    run_experiments.run_crosslingual(base_dataset, target_dataset, ifm_or_nifm, model=clf, k=kfolds)

if plot_results:
    if run_crosslingual:
        from plotting import results_visualised
        results_visualised.plot_TL_performance(base_dataset, target_dataset)
    if run_monolingual:
        #todo: monolingual plotting
        pass

if run_pretrained:
    import pretrained_model
    pretrained_model.run_ptm(base_dataset)

# from plotting import results_visualised
# results_visualised.plot_TL_performance(base_dataset, target_dataset)