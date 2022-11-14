# to create consistent validation set
# making the json file that contains various information about the corresponding data

# In this code, we use musdb_a_test as a validation set.
# For making 'unison of singing + singing' validation set.


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

    new_wav, sr = librosa.load(data_path, sr=None)
    new_wav_length = new_wav.shape[0]
    random_positions = random.randint(0, new_wav_length - (segment_length * sr))
    
    pitch_shift_ratio = random.uniform(-0.2, 0.2)  # -15 to +15 cents
    pitch_shift_ratio = random.choice([-12, 0, 12]) + pitch_shift_ratio
    formant_shift_ratio = random.uniform(1, 1.4)
    formant_shift_ratio = random.choice([formant_shift_ratio, 1 / formant_shift_ratio])
    pitch_range_ratio = random.uniform(1, 1.5)
    pitch_range_ratio = random.choice([pitch_range_ratio, 1 / pitch_range_ratio])
    time_stretch_ratio = max(0.7, random.normalvariate(1.0, 0.05))

    data1_data2_dict[basename] = {
        "corresponding_data": os.path.basename(data_path),
        "position(sec)": random_positions / sr,
        "unison_aug": True,
        "unison_params": [
            pitch_shift_ratio,
            formant_shift_ratio,
            pitch_range_ratio,
            time_stretch_ratio,
        ],
        "type": "singing_unison",
    }


with open("./valid_regions_dict_singing_unison.json", "w") as json_file:
    json.dump(data1_data2_dict, json_file, indent=0)
