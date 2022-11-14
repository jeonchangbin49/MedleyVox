# to create consistent validation set of singing + speech dataset
# making the json file that contains the starting positions of songs in validation audio data

# We use musdb_a_test and librispeech-dev as a validation set
# For making 'singing + speech'
import os
import glob
import json
import random
import tqdm

import librosa
import numpy as np

random.seed(777)

segment_length = 3  # 3 seconds

data_list = glob.glob("/path/to/data/24k/musdb_a_test/*.wav")
singing_data_list = glob.glob("/path/to/data/24k/LibriSpeech_dev-clean/*.wav")

data1_data2_dict = {}
for data_path in tqdm.tqdm(data_list):
    basename = os.path.basename(data_path)

    new_path = random.choice(singing_data_list)
    new_wav, sr = librosa.load(new_path, sr=None)

    too_short = True
    while too_short:
        if new_wav.shape[0] < segment_length * sr:
            new_path = random.choice(singing_data_list)
            new_wav, sr = librosa.load(new_path, sr=None)
        else:
            too_short = False
    random_positions = random.randint(0, new_wav.shape[0] - (segment_length * sr))
    data1_data2_dict[basename] = {
        "corresponding_data": os.path.basename(new_path),
        "position(sec)": random_positions / sr,
        "unison_aug": False,
        "unison_params": None,
        "type": "singing+speech",
    }


with open("./valid_regions_dict_singing_speech.json", "w") as json_file:
    json.dump(data1_data2_dict, json_file, indent=0)
