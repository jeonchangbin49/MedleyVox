# to create consistent validation set of speech + speech dataset
# making the json file that contains the starting positions of songs in validation audio data

# We use librispeech-dev as a validation set
# For making the 'speech + speech of same speaker'

import os
import glob
import json
import random
import tqdm

import librosa
import numpy as np

random.seed(777)

segment_length = 3  # 3 seconds

singing_data_list = glob.glob("/path/to/data/24k/LibriSpeech_dev-clean/*.wav")

data1_data2_dict = {}

speaker_list = []
speaker_wise_path_dict = {}
for data_path in tqdm.tqdm(singing_data_list):
    basename = os.path.basename(data_path)
    speaker_list.append(basename.split("-")[0])
speaker_list = list(set(speaker_list))

for speaker in speaker_list:
    speaker_wise_path_dict[speaker] = []

for data_path in tqdm.tqdm(singing_data_list):
    basename = os.path.basename(data_path)
    speaker_wise_path_dict[basename.split("-")[0]].append(data_path)

for data_path in tqdm.tqdm(singing_data_list):
    basename = os.path.basename(data_path)

    same_file = True
    while same_file:  # 같은 화자와 섞음.
        new_path = random.choice(speaker_wise_path_dict[basename.split("-")[0]])
        if data_path == new_path:
            pass
        else:
            same_file = False

    new_wav, sr = librosa.load(new_path, sr=None)

    if new_wav.shape[0] < segment_length * sr:
        new_wav = np.pad(new_wav, (0, segment_length * sr - new_wav.shape[0]))
        random_positions = 0
    else:
        random_positions = random.randint(0, new_wav.shape[0] - (segment_length * sr))
    data1_data2_dict[basename] = {
        "corresponding_data": os.path.basename(new_path),
        "position(sec)": random_positions / sr,
        "unison_aug": False,
        "unison_params": None,
        "type": "speech_same_speaker",
    }


with open("./valid_regions_dict_speech_speech_same_speaker.json", "w") as json_file:
    json.dump(data1_data2_dict, json_file, indent=0)
