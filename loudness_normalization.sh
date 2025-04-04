#This script normalizes audio files to a certain loudness level using the EBU R128 loudness normalization procedure.
# This script is using ffmpeg-normalize 1.26.1
# in_dir: directory containing recordings to be normalized
# out_dir: output directory containing recordings normalized.

# NeuroVoz tdu done
# Neurovoz ddk done
# Italian tdu done
# Italian ddk done
# PCGITA tdu done
# PCGITA ddk done

in_dir=C:\\Users\\INDYD\\Documents\\RAIVD_data\\CzechPD\\modified_records\\
temp_dir=C:\\Users\\INDYD\\Documents\\RAIVD_data\\NeuroVoz\\records_temp\\ # Some random empty folder suffices; can be removed afterwards
out_dir=C:\\Users\\INDYD\\Documents\\RAIVD_data\\CzechPD\\records_a_norm\\

for audio_File in "$in_dir"/*
do
	uttID=$(basename "$audio_File")
  temp_path=$temp_dir$uttID
	ffmpeg-normalize $audio_File -o $temp_path -f -ar 16000
  out_path=$out_dir$uttID
  echo $out_path
  ffmpeg -i $temp_path -af "highpass=f=100, lowpass=f=3000" $out_path -loglevel error
 
done
