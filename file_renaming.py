import os, re, shutil
from pydub import AudioSegment
import pandas as pd
from collections import defaultdict
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np

if os.getenv("COLAB_RELEASE_TAG"):  # colab
    data_dir = '/content/drive/My Drive/RAIVD_data/'
elif os.name == 'posix':  # linux
    data_dir = '/home/indy/Documents/RAIVD_data/'
elif os.name == 'nt':  # windows
    data_dir = "C:\\Users\INDYD\Documents\RAIVD_data\\"


############################### NeuroVoz ###############################

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

def get_files_per_task():
    subj_files_a = defaultdict(list)
    subj_files_ddk = defaultdict(list)
    subj_files_tdu = defaultdict(list)

    dir = data_dir + "NeuroVoz\\audios\\"
    for fname in os.listdir(dir):
        if fname.startswith('._'): # data set contains hidden files
            continue
        subj_name = fname[:2] + fname[-8:-4]
        f_path = os.path.join(dir, fname)
        if re.match(r"^[A-Z]{2}_A\d_\d+$", fname[:-4]):  # A
            subj_files_a[subj_name].append(f_path)
        elif 'pataka' in fname.lower():  # ddk
            subj_files_ddk[subj_name].append(f_path)
        elif 'espontanea' in fname.lower():  # spontaneous speech
            pass  # not used in this thesis
        elif len(fname[:-4]) > 11: # hardcoded: all filenames longer than 11 chars contains words instead of vowels
            subj_files_tdu[subj_name].append(f_path)
    return subj_files_a, subj_files_ddk, subj_files_tdu

def concat_files(subj_files, new_name, new_dir):
    for subj_id, files in subj_files.items():
        concat_audio = None
        for audio_file in files:
            audio = AudioSegment.from_wav(audio_file)
            if concat_audio is None:
                concat_audio = audio
            else: concat_audio += audio
        output_fname = os.path.join(new_dir, f"{subj_id[:2]}_{new_name}_{subj_id[2:]}")
        concat_audio.export(output_fname+'.wav', format='wav')

def rename_neurovoz_files():
    modified_dir = data_dir + "NeuroVoz\\records"
    new_dirs = [modified_dir+"\\", modified_dir+"_ddk\\", modified_dir+"_tdu\\"]
    subj_files_a, subj_files_ddk, subj_files_tdu = get_files_per_task()
    for files, name, dir in zip([subj_files_a, subj_files_ddk, subj_files_tdu], ['A1', 'DDK', 'TDU'], new_dirs):
        concat_files(files, name, dir)

############################### PC-GITA ###############################

def rename_pcgita_files():
    dir = data_dir + "PCGITA\\records_rp_norm\\"
    modified_dir = data_dir + "PCGITA\\records_readtxt_norm\\"
    subj_files = defaultdict(list)

    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if file.startswith('AVPEPUDEAC'):  # Healthy Control
                subject_id = 'HC_{:04d}'.format([int(s) for s in re.findall(r'\d+', file)][0]+100)
            elif file.startswith('AVPEPUDEA0'): # PD
                subject_id = 'PD_{:04d}'.format([int(s) for s in re.findall(r'\d+', file)][0])
            f_path = os.path.join(subdir, file)
            subj_files[subject_id].append(f_path)
    
    # To copy every recording separately
    for subj_id, files in subj_files.items():
        for i, audio_file in enumerate(files):
            output_fname = os.path.join(modified_dir, f"{subj_id[:2]}_RP{i+1}_{subj_id[-4:]}.wav")
            shutil.copy(audio_file, output_fname)
        
    # To concatenate audio recordings, use:
    # for subj_id, files in subj_files.items():
    #     concat_audio = None
    #     for audio_file in files:
    #         audio = AudioSegment.from_wav(audio_file)
    #         if concat_audio is None:
    #             concat_audio = audio
    #         else: concat_audio += audio
    #     output_fname = os.path.join(modified_dir, f"{subj_id[:2]}_DDK_{subj_id[-4:]}.wav")
    #     concat_audio.export(output_fname, format='wav')

rename_pcgita_files()


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


############################### CzechPD ###############################

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

def fake_gender():
    dataset = 'MDVR'
    # czech data is lacking gender, so create a file with all zeroes
    dir = data_dir + dataset + "\\records_tdu_norm\\"
    ids = []
    for file in os.listdir(dir):
        ids.append(file[-8:-4])
    ids = list(sorted(set(ids)))  # remove dups

    df = pd.DataFrame(data=ids, columns=['ID'])
    df['Sex'] = 0
    df.to_csv(os.path.join(data_dir, dataset, 'gender.csv'), index=False)

############################### IPVS ###############################

def get_gender(string):
    match = re.search(r'[a-zA-Z](?=[^a-zA-Z]*$)', string).group(0)
    return 1 if match == 'M' else 0


def get_files_per_task():
    data_folders = ['15 Young Healthy Control', '22 Elderly Healthy Control', '28 People with Parkinson\'s disease']
    status = ['HC', 'HC', 'PD']
    subj_files_a = defaultdict(list)
    subj_files_ddk = defaultdict(list)
    subj_files_tdu = defaultdict(list)
    genderinfo = defaultdict(list)
    id_list = defaultdict(list)
    new_id_count = 0

    for i, folder in enumerate(data_folders):
        dir = data_dir + "IPVS\\{}\\".format(folder)
        for subdir, dirs, files in os.walk(dir):
            
            id = subdir.split('\\')[-1]
            if id in id_list:
                new_id = id_list[id]
            else:
                new_id_count += 1
                new_id = "{:04d}".format(new_id_count)
                print(new_id_count, id)
                id_list[id] = new_id
            
            for file in files:
                subj_name = status[i] + new_id
                gender = get_gender(file[3:15])
                genderinfo[subj_name] = gender
                f_path = os.path.join(subdir, file)
                if file[1] == 'A': 
                    subj_files_a[subj_name].append(f_path)
                if file[0] == 'D': 
                    subj_files_ddk[subj_name].append(f_path)
                if file[:2] == 'FB':
                    subj_files_tdu[subj_name].append(f_path)
    return subj_files_a, subj_files_ddk, subj_files_tdu, genderinfo


def concat_files(subj_files, new_name, new_dir):
    for subj_id, files in subj_files.items():
        concat_audio = None
        for audio_file in files:
            audio = AudioSegment.from_wav(audio_file)
            if concat_audio is None:
                concat_audio = audio
            else: concat_audio += audio
        output_fname = os.path.join(new_dir, f"{subj_id[:2]}_{new_name}_{subj_id[2:]}")
        concat_audio.export(output_fname+'.wav', format='wav')

def rename_italian_files():
    modified_dir = data_dir + "IPVS\\records"
    new_dirs = [modified_dir+"\\", modified_dir+"_ddk\\", modified_dir+"_tdu\\"]
    subj_files_a, subj_files_ddk, subj_files_tdu, genderinfo = get_files_per_task()
    for files, name, dir in zip([subj_files_a, subj_files_ddk, subj_files_tdu], ['A1', 'DDK', 'TDU'], new_dirs):
        concat_files(files, name, dir)
    
    df = pd.DataFrame.from_dict(genderinfo.items())
    df.columns = ['ID', 'Sex']
    df = df.explode('Sex')
    df['ID'] = df['ID'].str[2:]
    df.to_csv(data_dir + 'IPVS\\gender.csv', index=False)

 ########################### Turkish #########################################

def read_turkish_df():
    df = pd.read_csv(os.path.join(data_dir, 'Turkish', 'pd_speech_features.csv'))  
    df.insert(len(df.columns)-1, 'gender', df.pop('gender'))
    df.insert(len(df.columns)-1, 'id', df.pop('id'))
    df = df.rename(columns={'class': 'y',
                            'id': 'subject_id'})
    df['sample_id'] = list(range(756))
    print(df.columns)
    # n=20
    # for i in range(0, len(df.columns), n ):
    #     print(df.columns[i:i+n])
    df.to_csv(os.path.join(data_dir, 'preprocessed_data', 'TurkishPD_tdu_norm_ifm.csv'), index=False)

############################### MDVR-KCL ###############################

def rename_mdvr():
    modified_dir = data_dir + "MDVR\\records_tdu"
    condition = ['PD', 'HC']
    for c in condition:
        audio_dir = data_dir  + "MDVR\\ReadText\\" + c
        for audio_file in os.listdir(audio_dir):
            audio = AudioSegment.from_wav(os.path.join(audio_dir,audio_file))
            audio_segment = audio[40000:100000]
            output_fname = os.path.join(modified_dir, f"{c}_TDU_00{audio_file[2:4]}")
            print(output_fname)
            audio_segment.export(output_fname+'.wav', format='wav')

############################### Other functions ###############################

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
    folder = 'IPVS'
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


def combine_rows(sets,d1,d2,d3=None):
    store_location = os.path.join(data_dir, 'preprocessed_data')
    df = pd.read_csv(os.path.join(store_location, f'{d1}{sets}.csv'))
    df2 = pd.read_csv(os.path.join(store_location, f'{d2}{sets}.csv'))
    df2['subject_id'] = df2.apply(lambda x: int(x['subject_id']) + 200, axis=1)
    df2['sample_id'] = df2.apply(lambda x: int(x['sample_id']) + 200, axis=1)
   
    if d3:
        df3 = pd.read_csv(os.path.join(store_location, f'{d3}{sets}.csv'))
        df3['subject_id'] = df3.apply(lambda x: int(x['subject_id']) + 400, axis=1)
        df3['sample_id'] = df3.apply(lambda x: int(x['sample_id']) + 400, axis=1)
        df_all = pd.concat([df, df2, df3])
        dfname = f'{d1}{d2}{d3}{sets}.csv'
    else:
        df_all = pd.concat([df, df2])
        dfname = f'{d1}{d2}{sets}.csv'
    print(dfname, df_all.columns, df_all.shape)
    df_all.to_csv(os.path.join(store_location,dfname), index=False)

# combine_rows('PCGITA', 'IPVS', 'MDVR', '_tdu_norm_vgg')

def combine_sets():
    for ds in [#('NeuroVoz', 'PCGITA', 'IPVS'), ('NeuroVoz', 'PCGITA', 'MDVR'), 
               #('NeuroVoz', 'IPVS', 'MDVR'), 
               ('PCGITA', 'IPVS')]: #'MDVR'
        for sets in ['_tdu', '_ddk']:
            for norm in ['_norm_ifm', '_norm_vgg']:
                combine_rows(sets+norm, ds[0], ds[1])


def combine_columns():
    store_location = os.path.join(data_dir, 'preprocessed_data')
    df = pd.read_csv(os.path.join(store_location, 'NeuroVoztdu_ifm.csv'))
    df2 = pd.read_csv(os.path.join(store_location, 'NeuroVoztdu_nifm.csv'))
    df.drop(['y', 'sample_id', 'gender'], inplace=True, axis=1)
    df_both = pd.merge(left=df, right=df2, on='subject_id')
    df_both.insert(len(df_both.columns)-3, 'subject_id', df_both.pop('subject_id'))

    print(df_both.columns, df_both.shape)
    df_both.to_csv(os.path.join(store_location,'NeuroVoztdu_both.csv'), index=False)


def preprocess():
    store_location = os.path.join(data_dir, 'preprocessed_data')
    df = pd.read_csv(os.path.join(data_dir, 'train_data.txt'))
    # df2 = pd.read_csv(os.path.join(data_dir, 'test_data.txt'))
    ran = list(range(26))
    print(ran)
    df.columns = ['subject_id'] +ran +['gender', 'y']
    df['sample_id'] = 0
    # df2.columns = ['subject_id'] +ran +['gender', 'y']
    # df2['sample_id'] = 0
    # df_both = pd.concat([df, df2])
    df.insert(len(df.columns)-3, 'subject_id', df.pop('subject_id'))


    print(df.columns, df.shape)
    df.to_csv(os.path.join(store_location,'sakar_a.csv'), index=False)


def getfeats():
    feats = [['dmfcc1_mean', 'dF0_mean', 'dmfcc9_mean', 'ddmfcc8_std', 'dmfcc8_std', 'dmfcc7_std', 'dmfcc11_std', 'ddmfcc4_std', 'dbShimmer', 'ddmfcc4_kurt', 'mfcc5_mean', 'ddmfcc9_std', 'ddmfcc11_std', 'APQ5', 'ddmfcc10_mean', 'APQ11', 'dmfcc4_std'],
['APQ3', 'DDA', 'APQ5', 'mfcc7_mean', 'dmfcc3_skew', 'mfcc9_mean', 'mfcc8_mean', 'mfcc6_mean', 'mfcc10_mean', '%Shimmer', 'mfcc9_std', 'mfcc4_mean', 'mfcc11_mean', 'mfcc1_mean', 'ddmfcc4_std', 'dbShimmer'],
['dmfcc1_mean', 'dmfcc2_mean', 'dF0_mean', 'dmfcc2_skew', 'dmfcc7_std', 'dmfcc9_mean', 'dmfcc8_std', 'dmfcc5_mean', 'APQ11', 'mfcc8_std', 'dmfcc4_std', 'dmfcc3_mean']
    ]
    # feats = [[('dmfcc13_skew', 0.0), ('ddmfcc2_skew', 0.0), ('ddmfcc5_skew', 0.0), ('ddmfcc6_skew', 0.0), ('ddmfcc7_skew', 0.0), ('ddmfcc8_skew', 0.0), ('ddmfcc9_skew', 0.0), ('ddmfcc10_skew', 0.0), ('ddmfcc11_skew', 0.0), ('ddmfcc12_skew', 0.0), ('ddmfcc13_skew', 0.0), ('mfcc2_kurt', 0.0), ('mfcc4_kurt', 0.0), ('mfcc5_kurt', 0.0), ('mfcc6_kurt', 0.0), ('mfcc8_kurt', 0.0), ('mfcc10_kurt', 0.0), ('mfcc11_kurt', 0.0), ('ddmfcc12_kurt', 0.0), ('ddmfcc13_kurt', 0.0)],
    #         [('ddmfcc7_skew', 0.0), ('ddmfcc8_skew', 0.0), ('ddmfcc9_skew', 0.0), ('ddmfcc10_skew', 0.0), ('ddmfcc11_skew', 0.0), ('ddmfcc12_skew', 0.0), ('ddmfcc13_skew', 0.0), ('mfcc2_kurt', 0.0), ('mfcc4_kurt', 0.0), ('mfcc5_kurt', 0.0), ('mfcc6_kurt', 0.0), ('mfcc7_kurt', 0.0), ('mfcc8_kurt', 0.0), ('mfcc9_kurt', 0.0), ('mfcc10_kurt', 0.0), ('mfcc11_kurt', 0.0), ('mfcc12_kurt', 0.0), ('mfcc13_kurt', 0.0), ('dmfcc10_kurt', 0.0), ('ddmfcc11_kurt', 0.0)],
    #         [('ddmfcc3_skew', 0.0), ('ddmfcc4_skew', 0.0), ('ddmfcc5_skew', 0.0), ('ddmfcc7_skew', 0.0), ('ddmfcc8_skew', 0.0), ('ddmfcc9_skew', 0.0), ('ddmfcc10_skew', 0.0), ('ddmfcc13_skew', 0.0), ('mfcc3_kurt', 0.0), ('mfcc4_kurt', 0.0), ('mfcc5_kurt', 0.0), ('mfcc6_kurt', 0.0), ('mfcc7_kurt', 0.0), ('mfcc9_kurt', 0.0), ('mfcc10_kurt', 0.0), ('mfcc11_kurt', 0.0), ('mfcc12_kurt', 0.0), ('mfcc13_kurt', 0.0), ('dmfcc12_kurt', 0.0), ('dmfcc13_kurt', 0.0)]
    #         ]
    from collections import Counter
    combined = [item for lst in feats for item in lst]  # Flatten all lists into one
    element_counts = Counter(combined)  # Count occurrences of each element
    common_elements = [item for item, count in element_counts.items() if count > 1]
    print(common_elements)

    dic = defaultdict(list)
    for f in feats:
        for f0, f1 in f:
            if f0 in dic.keys():
                dic[f0].append(f1)
            else:
                dic[f0] = [f1]
            print(f0, f1)
    print(dic)
    lis = []
    for k, v in dic.items():
        if len(v) > 1 or np.mean(v)>0.025:
            print(k, len(v), np.mean(v))
            lis.append(k)
    print(lis)


def add_dataset_col():
    store_location = os.path.join(data_dir, 'preprocessed_data')

    for file in os.listdir(store_location):
        dataset = file.split('_')[0]
        print(file, dataset)
        if len(dataset) > 10:  # skip concatenated datasets
            continue
        df = pd.read_csv(os.path.join(store_location, file))
        df['dataset'] = dataset
        print(df.shape, df.columns)
        df.to_csv(os.path.join(store_location, file), index=False)

def replace_filename():
    old = "SVM"
    new = 'SGD'
    cwd = os.path.abspath(os.getcwd())
    folder_path = os.path.join(cwd,'experiments')
    for file_name in os.listdir(folder_path):
        if old in file_name:
            new_name = file_name.replace(old, new)
            os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))
