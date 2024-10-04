import configparser

config = configparser.ConfigParser()
config.read('settings.ini')

base_dataset = config['DATA_SETTINGS']['base_dataset']
target_dataset = config['DATA_SETTINGS']['target_dataset']

ifm_or_nifm = config['EXPERIMENT_SETTINGS']['ifm_or_nifm']
file_or_window = config['DATA_SETTINGS']['file_or_window']

recreate_features = config.getboolean('RUN_SETTINGS', 'recreate_features')
run_models = config.getboolean('RUN_SETTINGS', 'run_models')
run_tl_models = config.getboolean('RUN_SETTINGS', 'run_tl_models')


plot_results = config.getboolean('OUTPUT_SETTINGS', 'plot_results')

if ifm_or_nifm == 'ifm':
    ifm_or_nifm += '_{}'.format(file_or_window)

if recreate_features:
    import get_features
    print("Creating {} features for '{}' dataset ".format(ifm_or_nifm, base_dataset))
    get_features.create_features(base_dataset, ifm_or_nifm)
    print("Creating {} features for '{}' dataset ".format(ifm_or_nifm, target_dataset))
    get_features.create_features(target_dataset, ifm_or_nifm)

if run_models:
    import ML_models
    print("Now running the {} ML model for '{}' dataset ".format(ifm_or_nifm, base_dataset))
    ML_models.run_ml_model(base_dataset, ifm_or_nifm)
    # models.run_cnn_model(dataset, ifm_or_nifm)

if run_tl_models:
    import ML_TL_model
    print("Now running the {} ML model with {}-base and {}-target data ".format(ifm_or_nifm, base_dataset, target_dataset))
    ML_TL_model.run_experiment(base_dataset, target_dataset, ifm_or_nifm)

if plot_results:
    from plotting import results_visualised
    results_visualised.plot_TL_performance(base_dataset, target_dataset)
