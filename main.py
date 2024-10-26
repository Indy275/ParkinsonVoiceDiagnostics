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
run_models = config.getboolean('RUN_SETTINGS', 'run_models')
run_tl_models = config.getboolean('RUN_SETTINGS', 'run_tl_models')

plot_results = config.getboolean('OUTPUT_SETTINGS', 'plot_results')

if speech_task == 'tdu':
    base_dataset += 'tdu'
    target_dataset += 'tdu'

if recreate_features:
    import get_features
    print("Creating {} features for '{}' dataset ".format(ifm_or_nifm, base_dataset))
    get_features.create_features(base_dataset, ifm_or_nifm)

    if not target_dataset.startswith('pass'):
        print("Creating {} features for '{}' dataset ".format(ifm_or_nifm, target_dataset))
        get_features.create_features(target_dataset, ifm_or_nifm)

if run_models:
    import run_experiments
    print("Now running the {} for '{}' dataset with {} features ".format(clf, base_dataset, ifm_or_nifm))
    run_experiments.run_monolingual(base_dataset, ifm_or_nifm, model=clf, k=kfolds)

if run_tl_models:
    import run_experiments
    print("Now running the {} ML model with {}-base and {}-target data ".format(ifm_or_nifm, base_dataset,
                                                                                target_dataset))
    run_experiments.run_crosslingual(base_dataset, target_dataset, ifm_or_nifm, model=clf, k=kfolds)

if plot_results:
    if run_tl_models:
        from plotting import results_visualised
        results_visualised.plot_TL_performance(base_dataset, target_dataset)
    if run_models:
        #todo: monolingual plotting
        pass
