import get_features, models

dataset = 'Czech'  # NeuroVoz Italian Czech test
ifm_or_nifm = 'ifm'
file_or_window = 'file'  # if True, get static file-level descriptors. If False, get window-level descriptors
recreate_features = True


if ifm_or_nifm == 'ifm':
    ifm_or_nifm += '_{}'.format(file_or_window)
if recreate_features:
    print("Creating {} features for '{}' dataset ".format(ifm_or_nifm, dataset))
    get_features.create_features(dataset, ifm_or_nifm)

print("Now running the ML model ".format(ifm_or_nifm, dataset))

models.run_ml_model(dataset, ifm_or_nifm)
