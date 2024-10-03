import configparser

config = configparser.ConfigParser()
config.read('settings.ini')

dataset = config['DATA_SETTINGS']['dataset']
ifm_or_nifm = config['DATA_SETTINGS']['ifm_or_nifm']
file_or_window = config['DATA_SETTINGS']['file_or_window']

recreate_features = config.getboolean('RUN_SETTINGS', 'recreate_features')
run_models = config.getboolean('RUN_SETTINGS', 'run_models')

if ifm_or_nifm == 'ifm':
    ifm_or_nifm += '_{}'.format(file_or_window)

if recreate_features:
    import get_features
    print("Creating {} features for '{}' dataset ".format(ifm_or_nifm, dataset))
    get_features.create_features(dataset, ifm_or_nifm)

if run_models:
    import ML_models
    print("Now running the ML model ".format(ifm_or_nifm, dataset))
    ML_models.run_ml_model(dataset, ifm_or_nifm)
    # models.run_cnn_model(dataset, ifm_or_nifm)
