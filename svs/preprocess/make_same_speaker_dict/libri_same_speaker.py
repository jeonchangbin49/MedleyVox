# make dict of LibriSpeech_train-clean-360, LibriSpeech_train-clean-100 dataset's {same singer: [song_list]}

import os
import glob
import json

train_list = glob.glob(
    "/path/to/data/24k/LibriSpeech_train-clean-360/*.wav"
)
total_list = train_list

singer_list = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("-")
    singer_name = "LibriSpeech_train-clean-360" + " - " + song_name_split[0]
    singer_list.append(singer_name)

singer_list = sorted(list(set(singer_list)))

song_path_dict = {}

for singer in singer_list:
    song_path_dict[singer] = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("-")
    singer_name = "LibriSpeech_train-clean-360" + " - " + song_name_split[0]
    song_path_dict[singer_name] += [song_name]

with open("./same_singer_LibriSpeech_train-clean-360.json", "w") as json_file:
    json.dump(song_path_dict, json_file, indent=0)


train_list = glob.glob(
    "/path/to/data/24k/LibriSpeech_train-clean-100/*.wav"
)
total_list = train_list

singer_list = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("-")
    singer_name = "LibriSpeech_train-clean-100" + " - " + song_name_split[0]
    singer_list.append(singer_name)

singer_list = sorted(list(set(singer_list)))

song_path_dict = {}

for singer in singer_list:
    song_path_dict[singer] = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("-")
    singer_name = "LibriSpeech_train-clean-100" + " - " + song_name_split[0]
    song_path_dict[singer_name] += [song_name]

with open("./same_singer_LibriSpeech_train-clean-100.json", "w") as json_file:
    json.dump(song_path_dict, json_file, indent=0)
