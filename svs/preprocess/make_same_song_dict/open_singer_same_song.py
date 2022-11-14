# check OpenSinger dataset's {same song: [song_list]}
# UNFORTUNATELY, this result does not guarantee a dictionary containing real same song T.T
# SO, WE WILL NOT USE THIS IN TRAINING.

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

song_list = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("_")
    song_name = song_name_split[1] + "_" + song_name_split[2]
    song_list.append(song_name)

song_list = sorted(list(set(song_list)))


song_path_dict = {}

for song in song_list:
    song_path_dict[song] = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("_")
    song_name = song_name_split[1] + "_" + song_name_split[2]
    man_or_woman = path.split("/")[-3]
    if song in song_list:
        basename = (
            os.path.basename(path).replace(".wav", "").split("_")[0]
            + "_"
            + os.path.basename(path).replace(".wav", "").split("_")[1]
            + "_"
            + os.path.basename(path).replace(".wav", "").split("_")[2]
        )
        song_path_dict[song_name] += [f"{man_or_woman}_{basename}"]

del_keys = []
for key, value in song_path_dict.items():
    if len(value) == 1:
        del_keys.append(key)
for key in del_keys:
    del song_path_dict[key]


with open("./same_song_dict_OpenSinger.json", "w") as json_file:
    json.dump(song_path_dict, json_file, indent=0)


# OpenSinger (Chinese)

# JVS-Music (Japanese)

# CSD (Korean, English)

# MedleyDB, MedleyDB 2.0 (English) => musdb song 제외


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

song_list = []
for path in total_list:
    song_name = os.path.basename(path).replace(".wav", "")
    song_name_split = song_name.split("_")
    song_name = song_name_split[1]
    song_list.append(song_name)

song_list = sorted(list(set(song_list)))

song_path_dict = {}

for song in song_list[:3]:
    singer_song_path_list = glob.glob(
        f"/path/to/data/OpenSinger/*/*_{song}"
    )
    if len(singer_song_path_list) == 1:
        pass
    else:
        print(song)
        song_lengths = []  # [26, 26, 26, 27]
        singer_song_names = []  # ['20_Dance', '23_Dance', '28_Dance', '35_Dance']
        for singer in singer_song_path_list:
            song_glob = glob.glob(f"{singer}/*.wav")
            song_lengths.append(len(song_glob))
            singer_song_names.append(os.path.basename(singer))

            # for i, singer_song_name in enumerate(singer_song_names):
            for i, song_length in enumerate(song_lengths):
                for l in range(song_length):
                    song_path_dict[
                        f"{os.path.basename(singer).split('_')[1]}_{l}_of_{song_lengths[i]}"
                    ] = []
        for singer in singer_song_path_list:
            for i, singer_song_name in enumerate(singer_song_names):
                singer_song_names_copy = singer_song_names.copy()
                song_lengths_copy = song_lengths.copy()
                del singer_song_names_copy[i]
                del song_lengths_copy[i]
                for k, copy_singer_song_name in enumerate(singer_song_names_copy):
                    if song_lengths[i] == song_lengths_copy[k]:
                        song_glob = glob.glob(f"{singer}/*.wav")
                        for j, file_path in enumerate(song_glob):
                            basename = os.path.basename(file_path).replace(
                                ".wav", ""
                            )  # "4_\u9ed8_1"
                            basename_split = basename.split("_")  # [4, \u9ed8, 1]
                            singer_name_total_of = f"{basename_split[1]}_{basename_split[2]}_of_{song_lengths[i]}"
                            song_path_dict[singer_name_total_of] += [
                                f"{singer.split('/')[-2]}_{basename}"
                            ]
