import os, re, shutil
import configparser
from pydub import AudioSegment
import pandas as pd
from collections import defaultdict
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np

config = configparser.ConfigParser()
config.read('settings.ini')
data_dir = config['DATA_SETTINGS']['data_dir']


def rename_neurovoz_ddk():
    dir = data_dir + "NeuroVoz\\DDK\\"
    modified_dir = data_dir + "NeuroVoz\\records_ddk\\"
    files = []
    for file in os.listdir(dir):
        files.append(file[:-4])
        output_fname = os.path.join(modified_dir, f"{file[:2]}_DDK_{file[-8:-4]}.wav")
        shutil.copy(os.path.join(dir, file), os.path.join(modified_dir, output_fname))
    print(len(files))
rename_neurovoz_ddk()

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

def rename_pcgita_files():
    dir = data_dir + "PC-GITA\\sentences\\"
    modified_dir = data_dir + "PC-GITA\\records_tdu\\"
    subj_files = defaultdict(list)

    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if file.startswith('AVPEPUDEAC'):  # Healthy Control
                subject_id = 'HC_{:04d}'.format([int(s) for s in re.findall(r'\d+', file)][0]+100)
            elif file.startswith('AVPEPUDEA0'): # PD
                subject_id = 'PD_{:04d}'.format([int(s) for s in re.findall(r'\d+', file)][0])
            f_path = os.path.join(subdir, file)
            subj_files[subject_id].append(f_path)
    print(len(subj_files))
    for k, v in subj_files.items():
        print(k, len(v))

    for subj_id, files in subj_files.items():
        concat_audio = None
        if len(files) > 1:
            for audio_file in files:
                audio = AudioSegment.from_wav(audio_file)
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
    match = re.search(r'[a-zA-Z](?=[^a-zA-Z]*$)', string).group(0)
    return 1 if match == 'M' else 0


def get_files_per_task():
    data_folders = ['15 Young Healthy Control', '22 Elderly Healthy Control', '28 People with Parkinson\'s disease']
    status = ['HC', 'HC', 'PD']
    subj_files_a = defaultdict(list)
    subj_files_ddk = defaultdict(list)
    subj_files_tdu = defaultdict(list)

    for i, folder in enumerate(data_folders):
        dir = data_dir + "ItalianPD\\{}\\".format(folder)
        for subdir, dirs, files in os.walk(dir):
            for file in files:
                subj_name = status[i] + file[3:15]
                f_path = os.path.join(subdir, file)
                if file[0] == 'D': 
                    subj_files_a[subj_name].append(f_path)
                if file[1] == 'A': 
                    subj_files_ddk[subj_name].append(f_path)
                if file[:2] == 'FB':
                    subj_files_tdu[subj_name].append(f_path)
    return subj_files_a, subj_files_ddk, subj_files_tdu


def concat_files(subj_files, new_name, new_dir):
    genderinfo = []
    new_id = 1
    for subj_id, files in subj_files.items():
        concat_audio = None
        for audio_file in files:
            audio = AudioSegment.from_wav(audio_file)
            if concat_audio is None:
                concat_audio = audio
            else: concat_audio += audio
        gender = get_gender(subj_id)
        genderinfo.append(('{:04d}'.format(new_id), gender))
        output_fname = os.path.join(new_dir, f"{subj_id[:2]}_{new_name}"+ "_{:04d}.wav".format(new_id))
        concat_audio.export(output_fname, format='wav')
        new_id += 1
    return genderinfo

def rename_italian_files():
    modified_dir = data_dir + "ItalianPD\\records"
    new_dirs = [modified_dir+"\\", modified_dir+"_ddk\\", modified_dir+"_tdu\\"]
    subj_files_a, subj_files_ddk, subj_files_tdu = get_files_per_task()
    genderinfo = []
    for files, name, dir in zip([subj_files_a, subj_files_ddk, subj_files_tdu], ['A1', 'DDK', 'TDU'], new_dirs):
        genderlist = concat_files(files, name, dir)
        genderinfo.extend(genderlist)
    genderinfo = np.array(genderinfo)
    genderDF = pd.DataFrame(genderinfo, columns=['ID', 'Sex'])
    genderDF.drop_duplicates(inplace=True)
    genderDF.to_csv(modified_dir + 'gender.csv', index=False)

rename_italian_files()

def neurovoz_sex():
    dir = data_dir + "NeuroVoz\\metadata\\"
    hc = pd.read_csv(dir + 'data_hc.csv', index_col=None, header=0)
    pwp = pd.read_csv(dir + 'data_pd.csv', index_col=None, header=0)

    df_1 = hc.loc[:, ['ID', 'Sex']]
    df_1['UPDRS scale'] = 0
    df_2 = pwp.loc[:, ['ID', 'UPDRS scale', 'Sex']]

    df = pd.concat([df_1, df_2])
    df.drop_duplicates(inplace=True)
    df['Sex'].fillna(1, inplace=True)
    df['ID'] = df['ID'].apply(lambda x: str(x).zfill(4))
    df.to_csv(dir + 'gender.csv', index=False)


def downsample(dataset, target_sample_rate=16000):
    modified_dir = data_dir + dataset
    for i, file in enumerate(os.listdir(modified_dir)):
        original_sample_rate, data = wavfile.read(modified_dir + file)

        number_of_samples = round(len(data) * float(target_sample_rate) / original_sample_rate)

        downsampled_data = resample(data, number_of_samples)

        downsampled_data = downsampled_data.astype(data.dtype)
        wavfile.write(modified_dir + file, target_sample_rate, downsampled_data)
        print("Downsampled file number", i)


def modify_df():
    folder = 'ItalianPD'
    store_location = os.path.join(data_dir, 'preprocessed_data')
    df = pd.read_csv(os.path.join(store_location, f'{folder}_ifm.csv'))
    print(df.columns)
    df.drop(columns='train_test', inplace=True)
    print(df.columns)
    df.to_csv(os.path.join(store_location, f'{folder}_ifm.csv'), index=False)


def modify_orig_id(row):
    if row['Indication']:
        return int(row['ORIG_ID']) + 25
    else:
        return int(row['ORIG_ID'])


def pcgita_gender():
    store_location = os.path.join(data_dir, 'PCGITA')
    df = pd.read_csv(os.path.join(store_location, 'PCGITA_metadata.csv'), header=0, delimiter=';', nrows=100)
    df.columns = df.columns.str.replace(' ', '')
    df['ORIG_ID'] = df['RECODINGORIGINALNAME'].str[-4:]
    df['Indication'] = df['RECODINGORIGINALNAME'].str[-5] == 'C'
    df.fillna(value=0, inplace=True)
    df['ID'] = df.apply(lambda x: int(x['ORIG_ID']) + 100 if x['Indication'] else int(x['ORIG_ID']), axis=1)
    df['ID'] = df['ID'].apply(lambda x: str(x).zfill(4))

    df['Sex'] = df.apply(lambda x: 1 if x['SEX']=='M' else 0, axis=1)
    df['Sex'] = df['SEX']

    df2 = df[['ID', 'Sex']]
    df2.to_csv(os.path.join(store_location,'gender.csv'), index=False)

def combine_dataframes():
    store_location = os.path.join(data_dir, 'preprocessed_data')
    df = pd.read_csv(os.path.join(store_location, 'ItalianPDtdu_ifm.csv'))
    df2 = pd.read_csv(os.path.join(store_location, 'NeuroVoztdu_ifm.csv'))
    # df3 = pd.read_csv(os.path.join(store_location, 'PCGITAtdu_ifm.csv'))

    df2['subject_id'] = df2.apply(lambda x: int(x['subject_id']) + 200, axis=1)
    df2['sample_id'] = df2.apply(lambda x: int(x['sample_id']) + 200, axis=1)

    # df3['subject_id'] = df3.apply(lambda x: int(x['subject_id']) + 400, axis=1)
    # df3['sample_id'] = df3.apply(lambda x: int(x['sample_id']) + 400, axis=1)

    df_all = pd.concat([df, df2])#, df3])
    print(df_all.columns, df_all.shape)
    df_all.to_csv(os.path.join(store_location,'ItalianPDNeuroVoztdu_ifm.csv'), index=False)
