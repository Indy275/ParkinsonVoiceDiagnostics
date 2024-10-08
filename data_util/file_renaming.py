import os, re, shutil
import configparser

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'settings.ini'))
data_dir = config['DATA_SETTINGS']['data_dir']


def rename_neurovoz_files():
    dir = data_dir + "NeuroVoz\\audios\\"
    modified_dir = data_dir + "NeuroVoz\\audios_A\\"
    files = []
    for file in os.listdir(dir):
        if re.match(r"^[A-Z]{2}_A\d_\d+$", file[:-4]):
            print("matched:", file)
            files.append(file[:-4])
            shutil.copy(os.path.join(dir, file), os.path.join(modified_dir, file))
    print(len(files))


def rename_czech_files():
    dir = data_dir + "CzechPD\\records\\"
    modified_dir = data_dir + "CzechPD\\modified_records\\"
    files = []
    for file in os.listdir(dir):
        if re.match(r"^[A-Z]{2}\d+a\d$", file[:-4]):
            files.append(file[:-4])
            if file[:2] == 'PD':
                new_fname = file[:2] + '_A' + file[-5] + '_{:04d}'.format(
                    int(''.join(re.findall(r'\d+', file[2:4]))) + 25) + '.wav'
            else:
                new_fname = file[:2] + '_A' + file[-5] + '_{:04d}'.format(
                    int(''.join(re.findall(r'\d+', file[2:4])))) + '.wav'
            shutil.copy(os.path.join(dir, file), os.path.join(modified_dir, new_fname))


def rename_italian_files():
    dir = data_dir + "ItalianPD\\15 Young Healthy Control\\"
    modified_dir = data_dir + "ItalianPD\\records\\"

    new_id = 56
    for root, dirs, files in os.walk(dir):
        new_id += 1
        for fname in files:
            if fname[1] == 'A':
                newname = 'HC_' + fname[1:3] + '_{:04d}'.format(new_id) + '.wav'
                print(fname, '-->', newname)
                shutil.copy(os.path.join(root, fname), os.path.join(modified_dir, newname))
