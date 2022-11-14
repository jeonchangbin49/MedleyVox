# make dict of jvs_music_ver1 dataset's {same singer: [song_list]}

import glob
import json

train_list = glob.glob(
    "/path/to/data/jvs_music_ver1/*/*/wav/raw.wav"
)
total_list = train_list

singer_list = []
for path in total_list:
    path_split = path.split("/")
    singer_name = path_split[-4]
    singer_name = "jvs_music_ver1" + " - " + singer_name
    singer_list.append(singer_name)

singer_list = sorted(list(set(singer_list)))

song_path_dict = {}

for singer in singer_list:
    song_path_dict[singer] = []
for path in total_list:
    path_split = path.split("/")
    unique_or_common = path_split[-3].replace("song_", "")
    singer_name = path_split[-4]
    singer_name = "jvs_music_ver1" + " - " + singer_name
    song_path_dict[singer_name] += [path_split[-4] + "_" + unique_or_common]

with open("./same_singer_jvs_music_ver1.json", "w") as json_file:
    json.dump(song_path_dict, json_file, indent=0)
