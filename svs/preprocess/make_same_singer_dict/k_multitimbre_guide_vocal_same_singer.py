# make dict of 다음색 가이드보컬 dataset's {same singer: [song_list]}

import os
import glob
import json

train_list = glob.glob(
    "/path/to/data/177.k_multitimbre_guide_vocal/01.data/1.Training/original_data/*/*/*/*/*/*.wav"
)
valid_list = glob.glob(
    "/path/to/data/177.k_multitimbre_guide_vocal/01.data/2.Validation/original_data/*/*/*/*/*/*.wav"
)

total_list = train_list + valid_list

singer_list = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("_")
    singer_name = (
        "k_multitimbre" + " - " + song_name_split[0] + "_" + song_name_split[1]
    )
    singer_list.append(singer_name)

singer_list = sorted(list(set(singer_list)))

song_path_dict = {}

for singer in singer_list:
    song_path_dict[singer] = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("_")
    singer_name = (
        "k_multitimbre" + " - " + song_name_split[0] + "_" + song_name_split[1]
    )
    song_path_dict[singer_name] += [song_name]

with open("./same_singer_k_multitimbre.json", "w") as json_file:
    json.dump(song_path_dict, json_file, indent=0)
