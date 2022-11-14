# make dict of kiritan dataset's {same singer: [song_list]}
# kiritan has only one singer
# however, we splitted one file to segments because the silence regions were too long.
# {'kiritan': [song_name]}

import soundfile as sf
import os
import glob
import tqdm
import json

preprocessed_list = glob.glob("/path/to/data/24k/kiritan/*.wav")

path_list = glob.glob("/path/to/data/kiritan/wav/*.wav")

total_list = path_list

singer_list = ["kiritan"]

song_path_dict = {}

for singer in singer_list:
    song_path_dict[singer] = []
for path in tqdm.tqdm(preprocessed_list):
    song_name = os.path.basename(path).replace(".wav", "")
    song_path_dict["kiritan"] += [song_name]

with open("./same_singer_kiritan.json", "w") as json_file:
    json.dump(song_path_dict, json_file, indent=0)
