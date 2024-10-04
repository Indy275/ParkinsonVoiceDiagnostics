import os
import re
import configparser

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'settings.ini'))
data_dir = config['DATA_SETTINGS']['data_dir']


def get_dirs(dataset):
    if dataset.lower() == 'neurovoz':
        folder = 'NeuroVoz'
        dir = data_dir + folder + '\\audios_A\\'
    elif dataset.lower() == 'czechpd':
        folder = 'CzechPD'
        dir = data_dir + folder + "\\modified_records\\"
    elif dataset.lower() == 'italianpd':
        folder = 'ItalianPD'
        dir = data_dir + folder + "\\records\\"
    elif dataset.lower() == 'test':
        folder = 'test'
        dir = data_dir + "\\NeuroVoz\\subsample\\"
    else:
        print(" '{}' is not a valid data set ".format(dataset))
        return
    store_location = data_dir + 'preprocessed_data\\{}_preprocessed\\'.format(folder)

    if not os.path.exists(store_location):
        os.makedirs(store_location)
    return dir, store_location


def load_files(datadir):
    files = []
    for file in os.listdir(datadir):
        if re.match(r".*^[A-Z]{2}_A\d_\d+$", file[:-4]):
            files.append(file[:-4])

    HC_id_list = [f[-4:] for f in files if f[:2] == 'HC']
    PD_id_list = [f[-4:] for f in files if f[:2] == 'PD']
    return files, list(set(HC_id_list)), list(set(PD_id_list))
