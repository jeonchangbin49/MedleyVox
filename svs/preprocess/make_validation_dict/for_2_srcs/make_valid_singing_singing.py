# to create consistent validation set
# making the json file that contains the starting positions of songs in validation audio data

# In this code, we use musdb_a_test as a validation set.
# For making 'different singing + singing' validation set.


import os
import glob
import json
import random
import tqdm

import librosa

random.seed(777)

segment_length = 3

data_list = glob.glob("/path/to/data/24k/musdb_a_test/*.wav")

data1_data2_dict = {}
for data_path in tqdm.tqdm(data_list):
    basename = os.path.basename(data_path)
    same_name = True
    while same_name:  # only applicable to musdb dataset
        new_path = random.choice(data_list)
        if basename.split(" - ")[0] != os.path.basename(new_path.split(" - ")[0]):
            same_name = False
        else:
            pass
    new_wav, sr = librosa.load(new_path, sr=None)
    new_wav_length = new_wav.shape[0]
    random_positions = random.randint(0, new_wav_length - (segment_length * sr))
    data1_data2_dict[basename] = {
        "corresponding_data": os.path.basename(new_path),
        "position(sec)": random_positions / sr,
        "unison_aug": False,
        "unison_params": None,
        "type": "singing+singing",
    }


with open("./valid_regions_dict_singing_singing.json", "w") as json_file:
    json.dump(data1_data2_dict, json_file, indent=0)
