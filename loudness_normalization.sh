#This script normalizes audio files to a certain loudness level using the EBU R128 loudness normalization procedure.
# This script is using ffmpeg-normalize 1.26.1
# in_dir: directory containing recordings to be normalized
# out_dir: output directory containing recordings normalized.

# NeuroVoz tdu
# Neurovoz ddk
# Italian tdu
# Italian ddk
# PCGITA tdu
# PCGITA ddk

in_dir=C:\\Users\\INDYD\\Documents\\RAIVD_data\\NeuroVoz\\records_tdu\\
out_dir=C:\\Users\\INDYD\\Documents\\RAIVD_data\\NeuroVoz\\records_tdu_norm\\

for audio_File in "$in_dir"/*
do
	uttID=$(basename "$audio_File")
  out_path=$out_dir$uttID
  echo $out_path
	ffmpeg-normalize $audio_File -o $out_path -f -ar 16000
done