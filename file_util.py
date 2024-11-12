import os
import re
import configparser

config = configparser.ConfigParser()
config.read('settings.ini')
speech_task = config['DATA_SETTINGS']['speech_task']
normalize_audio = config.getboolean('DATA_SETTINGS', 'normalize_audio')

if os.getenv("COLAB_RELEASE_TAG"):  # colab
    data_dir = '/content/drive/My Drive/RAIVD_data/'
elif os.name == 'posix':  # linux
    data_dir = '/home/indy/Documents/RAIVD_data/'
elif os.name == 'nt':  # windows
    data_dir = "C:\\Users\INDYD\Documents\RAIVD_data\\"


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
    elif dataset.lower().startswith('pcgita'):
        folder = 'PCGITA'
        dir = os.path.join(data_dir , folder, 'records')
    elif dataset.lower() == 'test':
        folder = 'test'
        dir = os.path.join(data_dir , 'NeuroVoz', 'subsample')
    else:
        folder = dataset[:-3]
        dir = os.path.join(data_dir , folder, 'records')

    if speech_task == 'tdu':
        dir = os.path.join(data_dir , folder, 'records_tdu')
        folder += 'tdu'
    elif speech_task == 'ddk':
        dir = os.path.join(data_dir , folder, 'records_ddk')
        folder += 'ddk'
    elif speech_task == 'lr':
        dir = os.path.join(data_dir , folder, 'records_lr')
        folder += 'lr'
    
    if normalize_audio:
        dir += '_norm'
        folder += '_norm'

    store_location = (os.path.join(data_dir, 'preprocessed_data'), folder)
    print("Load data from dir:", dir, store_location)
    return dir, store_location


def load_files(datadir):
    files = []
    if speech_task == 'ddk':
        task = 'DDK'
    elif speech_task == 'tdu': 
        task = 'TDU'
    else:
        task = r'A\d'
    
    for file in os.listdir(datadir):
        # if re.match(r".*^[A-Z]{2}_"+task+"_\d+$", file[:-4]):
            files.append(file[:-4])

    HC_id_list = [f[-4:] for f in files if f[:2] == 'HC']
    PD_id_list = [f[-4:] for f in files if f[:2] == 'PD']
    return files, list(set(HC_id_list)), list(set(PD_id_list))
