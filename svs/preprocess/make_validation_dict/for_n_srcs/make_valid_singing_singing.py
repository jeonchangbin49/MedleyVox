# to create consistent validation set
# making the json file that contains the starting positions of songs in validation audio data

# In this code, we use musdb_a_test as a validation set.
# For making 'different singing + singing + singing + singing + ...' validation set.

import os
import glob
import json
import random
import tqdm
import argparse

import librosa

parser = argparse.ArgumentParser(description="dataset parameter")
parser.add_argument(
    "--singing_data_root",
    type=str,
    default="/path/to/data/24k/musdb_a_test",
)
parser.add_argument("--sample_rate", type=int, default=24000)
parser.add_argument(
    "--seq_len", type=float, default=6, help="Sequence duration in seconds"
)
parser.add_argument(
    "--min_n_src", type=int, default=3, help="minimum number of sources"
)
parser.add_argument(
    "--max_n_src", type=int, default=10, help="maximum number of sources"
)

args, _ = parser.parse_known_args()

random.seed(777)

segment_length = args.seq_len

singing_data_list = glob.glob(f"{args.singing_data_root}/*.wav")

data1_data2_dict = {}
for data_path in tqdm.tqdm(singing_data_list):
    basename = os.path.basename(data_path)

    corresponding_data_list = []
    position_list = []
    gain_adjustment_list = []
    n_srcs = random.randint(args.min_n_src, args.max_n_src)

    for src in range(n_srcs - 1):
        same_name = True
        while same_name:  # only applicable to musdb dataset
            new_path = random.choice(singing_data_list)
            if basename.split(" - ")[0] != os.path.basename(new_path.split(" - ")[0]):
                same_name = False
            else:
                pass
        new_wav, sr = librosa.load(new_path, sr=None)
        if new_wav.shape[0] < segment_length * sr:
            random_positions = 0  # if wav is too short, just start from 0
        else:
            random_positions = random.randint(
                0, new_wav.shape[0] - (segment_length * sr)
            )
        # data1_data2_dict[basename] = [os.path.basename(new_path), random_positions / sr]
        corresponding_data_list.append(os.path.basename(new_path))
        position_list.append(random_positions / sr)
        gain_adjustment_list.append(random.uniform(0.25, 1.25))
    data1_data2_dict[basename] = {
        "corresponding_data": corresponding_data_list,
        "position(sec)": position_list,
        "gain_adjustment": gain_adjustment_list,
        "unison_aug": False,
        "unison_params": None,
        "type": "singing+singing",
    }


with open("./valid_regions_dict_singing_singing_n_srcs.json", "w") as json_file:
    json.dump(data1_data2_dict, json_file, indent=0)
