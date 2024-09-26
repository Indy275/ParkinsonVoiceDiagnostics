import os
import re

####### NeuroVoz ######

def load_files(datadir):
    files = []
    start_int = 2 if datadir=='neurovoz' else 0
    for file in os.listdir(datadir):
        if re.match(r".*^[A-Z]{2}_A\d_\d+$", file[start_int:-4]):
            files.append(file[start_int:-4])

    HC_id_list = [f[-4:] for f in files if f[:2] == 'HC']
    PD_id_list = [f[-4:] for f in files if f[:2] == 'PD']
    return files, list(set(HC_id_list)), list(set(PD_id_list))

def load_czech_files():
    # Deprecated; all files are now renamed
    dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\CzechPD\\modified_records\\"

    files = []
    for file in os.listdir(dir):
        if re.match(r"^[A-Z]{2}\d+a\d$", file[:-4]):
            files.append(file[:-4])

    HC_id_list = [f[2:4] for f in files if f[:2] == 'HC']
    PD_id_list = [f[2:4] for f in files if f[:2] == 'PD']

    HC_cleaned_id_list, PD_cleaned_id_list = [], []
    for it in HC_id_list:
        integer = re.findall(r'\d+', it)
        if integer:
            HC_cleaned_id_list.append(int(''.join(integer))+50)
    for it in PD_id_list:
        integer = re.findall(r'\d+', it)
        if integer:
            PD_cleaned_id_list.append(int(''.join(integer))+50)

    return files, list(set(HC_cleaned_id_list)), list(set(PD_cleaned_id_list))

def load_italian_files():
    return