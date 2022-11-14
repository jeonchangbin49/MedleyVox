# coding: utf8
# make dict of CSD dataset's {same singer: [song_list]}
# {singer: ["WomanRaw_4_\u9ed8_28"]}

import os
import glob
import json

song_list = glob.glob("/path/to/data/CSD/*/wav/*.wav")

total_list = song_list

singer_list = ["CSD"]

song_path_dict = {}

for singer in singer_list:
    song_path_dict[singer] = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_path_dict["CSD"] += [song_name]

with open("./same_singer_CSD.json", "w") as json_file:
    json.dump(song_path_dict, json_file, indent=0)
