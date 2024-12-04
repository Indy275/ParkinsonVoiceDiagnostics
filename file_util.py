import os
import re
import configparser

config = configparser.ConfigParser()
config.read('settings.ini')
normalize_audio = config.getboolean('DATA_SETTINGS', 'normalize_audio')

if os.getenv("COLAB_RELEASE_TAG"):  # colab
    data_dir = '/content/drive/My Drive/RAIVD_data/'
elif os.name == 'posix':  # linux
    data_dir = '/home/indy/Documents/RAIVD_data/'
elif os.name == 'nt':  # windows
    data_dir = "C:\\Users\INDYD\Documents\RAIVD_data\\"


def get_dirs(dataset):
    dir = os.path.join(data_dir , dataset.split('_')[0], f'records_'+dataset.split('_')[-1])
    
    if normalize_audio:
        dir += '_norm'
        dataset += '_norm'

    store_location = (os.path.join(data_dir, 'preprocessed_data'), dataset)
    print("Load data from dir:", dir, " and store to dir:", store_location)
    return dir, store_location


def load_files(datadir):
    files = []
    for file in os.listdir(datadir):
            files.append(file[:-4])

    HC_id_list = [f[-4:] for f in files if f[:2] == 'HC']
    PD_id_list = [f[-4:] for f in files if f[:2] == 'PD']
    return files, list(set(HC_id_list)), list(set(PD_id_list))
