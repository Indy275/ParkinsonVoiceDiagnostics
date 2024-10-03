import os
import re

####### NeuroVoz ######

def load_files(datadir):
    files = []
    for file in os.listdir(datadir):
        if re.match(r".*^[A-Z]{2}_A\d_\d+$", file[:-4]):
            files.append(file[:-4])

    HC_id_list = [f[-4:] for f in files if f[:2] == 'HC']
    PD_id_list = [f[-4:] for f in files if f[:2] == 'PD']
    return files, list(set(HC_id_list)), list(set(PD_id_list))
