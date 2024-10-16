import os
import re
import configparser

config = configparser.ConfigParser()
config.read('settings.ini')
data_dir = config['DATA_SETTINGS']['data_dir']
speech_task = config['DATA_SETTINGS']['speech_task']


def get_dirs(dataset):
    if dataset.lower().startswith('neurovoz'):
        folder = 'NeuroVoz'
        dir = os.path.join(data_dir , folder, 'audios_A')
    elif dataset.lower() == 'czechpd':
        folder = 'CzechPD'
        dir = os.path.join(data_dir , folder, 'modified_records')
    elif dataset.lower().startswith('italianpd'):
        folder = 'ItalianPD'
        dir = os.path.join(data_dir , folder, 'records')
    elif dataset.lower() == 'test':
        folder = 'test'
        dir = os.path.join(data_dir , 'NeuroVoz', 'subsample')
    else:
        print(" '{}' is not a valid data set ".format(dataset))
        return
    if speech_task == 'tdu':
        dir = os.path.join(data_dir , folder, 'records_tdu')
        folder = folder + 'tdu'
    store_location = os.path.join(data_dir, 'preprocessed_data',f'{folder}_preprocessed')

    if not os.path.exists(store_location):
        os.makedirs(store_location)
    return dir, store_location


def load_files(datadir):
    files = []
    task = 'TDU' if speech_task == 'tdu' else r'A\d'
    for file in os.listdir(datadir):
        if re.match(r".*^[A-Z]{2}_"+task+"_\d+$", file[:-4]):
            files.append(file[:-4])

    HC_id_list = [f[-4:] for f in files if f[:2] == 'HC']
    PD_id_list = [f[-4:] for f in files if f[:2] == 'PD']
    return files, list(set(HC_id_list)), list(set(PD_id_list))
