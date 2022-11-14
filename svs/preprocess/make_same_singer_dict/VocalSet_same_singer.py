# make dict of VocalSet dataset's {same singer: [song_list]}
# some files are not preprocessed, so we should consider an exception
# files that doesn't start with name 'f2_....' (f2 => female2) are not preprocessed


import os
import glob
import json

train_list = glob.glob(
    "/path/to/data/VocalSet/FULL/*/*/*/*.wav"
)
# valid_list = glob.glob('/path/to/data/004.k_multisinger/01.data/2.Validation/original_data/*/*/*/*/*.wav')

total_list = train_list

sex_index_dict = {}
for i in range(12):
    sex_index_dict[f"male{i}"] = f"m{i}"
    sex_index_dict[f"female{i}"] = f"f{i}"

singer_list = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("_")
    if song_name_split[0] == sex_index_dict[path.split("/")[-4]]:
        singer_name = "VocalSet" + " - " + path.split("/")[-4]
        singer_list.append(singer_name)
    else:
        pass

singer_list = sorted(list(set(singer_list)))

song_path_dict = {}

for singer in singer_list:
    song_path_dict[singer] = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("_")
    if song_name_split[0] == sex_index_dict[path.split("/")[-4]]:
        singer_name = "VocalSet" + " - " + path.split("/")[-4]
        song_path_dict[singer_name] += [song_name]
    else:
        pass

with open("./same_singer_VocalSet.json", "w") as json_file:
    json.dump(song_path_dict, json_file, indent=0)
