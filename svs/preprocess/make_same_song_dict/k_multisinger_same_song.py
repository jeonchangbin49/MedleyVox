# make dict of k_multisinger dataset's {same song: [song_list]}

import os
import glob
import json
import librosa
import tqdm

train_list = glob.glob(
    "/path/to/data/004.k_multisinger/01.data/1.Training/original_data/*/*/*/*/*.wav"
)
valid_list = glob.glob(
    "/path/to/data/004.k_multisinger/01.data/2.Validation/original_data/*/*/*/*/*.wav"
)

total_list = train_list + valid_list

song_list = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("_")
    real_song_name = song_name_split[1] + "_" + song_name_split[2]
    song_list.append(real_song_name)

song_list = sorted(list(set(song_list)))

song_path_dict = {}

for song in song_list:
    song_path_dict[song] = []
for path in tqdm.tqdm(total_list):
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("_")
    real_song_name = song_name_split[1] + "_" + song_name_split[2]
    wav, sr = librosa.load(path, sr=None)
    song_path_dict[real_song_name] += [[song_name, wav.shape[0] / sr]]

del_keys = []
for key, value in song_path_dict.items():
    if len(value) == 1:
        del_keys.append(key)
for key in del_keys:
    del song_path_dict[key]

with open("./same_song_k_multisinger.json", "w") as json_file:
    json.dump(song_path_dict, json_file, indent=0)
