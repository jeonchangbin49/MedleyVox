import os
import json
import glob
import argparse

import tqdm
import torchaudio

parser = argparse.ArgumentParser(description="dataset parameter")
parser.add_argument(
    "--sample_rate",
    type=int,
    default=24000,
    help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
)
parser.add_argument("--samplerate_folder_name", type=str, default="24k")

args, _ = parser.parse_known_args()


song_list = glob.glob(
    f"/path/to/data/{args.samplerate_folder_name}/*/*.wav"
)

song_length_dict = {}
for path in tqdm.tqdm(song_list):
    song_name = os.path.basename(path)
    length = torchaudio.info(path).num_frames
    song_length_dict[song_name] = length

with open(f"./song_length_dict_{args.samplerate_folder_name}.json", "w") as json_file:
    json.dump(song_length_dict, json_file, indent=0)


lengths = []
for key, item in song_length_dict.items():
    lengths.append(item)
print(f"total data lengths : {sum(lengths) / (args.sample_rate * 60)} mins")


# to check each dataset's total song lengths
song_list = glob.glob(
    f"/path/to/data/24k/OpenSinger/*.wav"
)

song_length_dict = {}
for path in tqdm.tqdm(song_list):
    song_name = os.path.basename(path)
    length = torchaudio.info(path).num_frames
    song_length_dict[song_name] = length

lengths = []
for key, item in song_length_dict.items():
    lengths.append(item)
print(f"total data lengths : {sum(lengths) / (24000 * 60)} mins")
