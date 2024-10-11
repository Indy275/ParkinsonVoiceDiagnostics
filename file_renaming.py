import os, re, shutil
import configparser
from pydub import AudioSegment
import pandas as pd
from collections import defaultdict
from scipy.io import wavfile
from scipy.signal import resample

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


def rename_neurovoz_files_tdu():
    dir = data_dir + "NeuroVoz\\audios\\"
    modified_dir = data_dir + "NeuroVoz\\records_tdu\\"
    subj_files = defaultdict(list)
    for file in os.listdir(dir):
        fname = file[:-4]
        if len(fname) > 10:  # Match all files except sustained vowel task
            if fname[0:2] in ['HC', 'PD']:
                subject_id = fname[0:2] + fname[-4:]
                subj_files[subject_id].append(fname)

    for subj_id, files in subj_files.items():
        concat_audio = None
        for audio_file in files:
            audio = AudioSegment.from_wav(os.path.join(dir, audio_file)+'.wav')
            if concat_audio is None:
                concat_audio = audio
            else: concat_audio += audio
        output_fname = os.path.join(modified_dir, f"{subj_id[:2]}_TDU_{subj_id[-4:]}.wav")
        concat_audio.export(output_fname, format='wav')


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


def get_gender(string):
    match = re.search(r'[a-zA-Z](?=[^a-zA-Z]*$)', string[:-4]).group(0)
    return 1 if match == 'M' else 0


def rename_italian_files():
    modified_dir = data_dir + "ItalianPD\\records_tdu\\"
    data_folders = ['15 Young Healthy Control', '22 Elderly Healthy Control', '28 People with Parkinson\'s disease']
    status = ['HC', 'HC', 'PD']
    new_id = 0
    genderinfo = []
    for i, folder in enumerate(data_folders):
        dir = data_dir + "ItalianPD\\{}\\".format(folder)
        for root, dirs, files in os.walk(dir):
            new_id += 1
            for fname in files:
                # if fname[1] == 'A':
                if fname[:2] == 'FB':
                    gender = get_gender(fname)
                    # fname[1:3]
                    newname = '{}_'.format(status[i]) + 'TDU'+ '_{:04d}'.format(new_id) + '.wav'
                    print(fname, '-->', newname)
                    shutil.copy(os.path.join(root, fname), os.path.join(modified_dir, newname))
                    genderinfo.append(('{:04d}'.format(new_id), gender))
    genderDF = pd.DataFrame(genderinfo, columns=['ID', 'Sex'])
    genderDF.drop_duplicates(inplace=True)
    genderDF.to_csv(modified_dir + 'gender.csv', index=False)


def neurovoz_sex():
    dir = data_dir + "NeuroVoz\\metadata\\"
    hc = pd.read_csv(dir + 'data_hc.csv', index_col=None, header=0)
    pwp = pd.read_csv(dir + 'data_pd.csv', index_col=None, header=0)

    df_1 = hc.loc[:, ['ID', 'Sex']]
    df_2 = pwp.loc[:, ['ID', 'Sex']]

    df = pd.concat([df_1, df_2])
    df.drop_duplicates(inplace=True)
    df['Sex'].fillna(1, inplace=True)
    df['ID'] = df['ID'].apply(lambda x: str(x).zfill(4))
    df.to_csv(dir + 'gender.csv', index=False)

neurovoz_sex()


def downsample(target_sample_rate=16000):
    modified_dir = data_dir + "ItalianPD\\records_tdu\\"
    for i, file in enumerate(os.listdir(modified_dir)):
        original_sample_rate, data = wavfile.read(modified_dir + file)

        number_of_samples = round(len(data) * float(target_sample_rate) / original_sample_rate)

        downsampled_data = resample(data, number_of_samples)

        downsampled_data = downsampled_data.astype(data.dtype)
        wavfile.write(modified_dir + file, target_sample_rate, downsampled_data)
        print("Downsampled file number", i)
