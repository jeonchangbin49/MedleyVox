# make dict of OpenSinger dataset's {same singer: [song_list]}
# {singer: ["WomanRaw_4_\u9ed8_28"]}

import os
import glob
import json

man_list = glob.glob(
    "/path/to/data/OpenSinger/ManRaw/*/*.wav"
)
woman_list = glob.glob(
    "/path/to/data/OpenSinger/WomanRaw/*/*.wav"
)

total_list = man_list + woman_list

singer_list = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("_")
    singer_name = "OpenSinger" + " - " + path.split("/")[-3] + "_" + song_name_split[0]
    singer_list.append(singer_name)

singer_list = sorted(list(set(singer_list)))

song_path_dict = {}

for singer in singer_list:
    song_path_dict[singer] = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("_")
    singer_name = "OpenSinger" + " - " + path.split("/")[-3] + "_" + song_name_split[0]
    song_path_dict[singer_name] += [path.split("/")[-3] + "_" + song_name]

with open("./same_singer_OpenSinger.json", "w") as json_file:
    json.dump(song_path_dict, json_file, indent=0)
