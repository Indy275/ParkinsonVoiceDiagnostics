import os, re, shutil


def rename_czech_files():
    dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\CzechPD\\records\\"
    modified_dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\CzechPD\\modified_records\\"

    files = []
    for file in os.listdir(dir):
        if re.match(r"^[A-Z]{2}\d+a\d$", file[:-4]):
            files.append(file[:-4])
            new_fname = file[:2] + '_A' + file[-5] +'_{:04d}'.format(int(''.join(re.findall(r'\d+', file[2:4])))) + '.wav'
            shutil.copy(os.path.join(dir, file), os.path.join(modified_dir, new_fname) )


def rename_italian_files():
    dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\ItalianPD\\15 Young Healthy Control\\"
    modified_dir = "C:\\Users\\INDYD\\Documents\\RAIVD_data\\ItalianPD\\records\\"

    new_id = 56
    for root, dirs, files in os.walk(dir):
        new_id += 1
        for fname in files:
            if fname[1] == 'A':
                newname = 'HC_' + fname[1:3] + '_{:04d}'.format(new_id) + '.wav'
                print(fname, '-->', newname)
                shutil.copy(os.path.join(root, fname), os.path.join(modified_dir, newname))
