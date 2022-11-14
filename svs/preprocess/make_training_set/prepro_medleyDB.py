# Due to the behavior of 'librosa.effects.split', I had to manually remove the small size files that created from this code.
# 

import os
import glob
import argparse
import yaml

import librosa
import soundfile as sf
import tqdm


parser = argparse.ArgumentParser(description="dataset parameter")
parser.add_argument(
    "--sample_rate",
    type=int,
    default=24000,
    help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
)
parser.add_argument("--samplerate_folder_name", type=str, default="24k")

args, _ = parser.parse_known_args()


musdb_path_list = glob.glob(
    f"/path/to/data/{args.samplerate_folder_name}/musdb_a_train/*.wav"
) + glob.glob(
    f"/path/to/data/{args.samplerate_folder_name}/musdb_a_test/*.wav"
)
musdb_list = []
for musdb_path in musdb_path_list:
    song_name = os.path.basename(musdb_path).replace(".wav", "")
    musdb_list.append(song_name.split("_")[0].split(" - ")[1].replace(" ", ""))
musdb_list = list(set(musdb_list))


# VOCALS = ["male singer", "female singer", "male speaker", "female speaker",
#           "male rapper", "female rapper", "beatboxing", "vocalists"]
VOCALS = [
    "male singer",
    "female singer",
    "male speaker",
    "female speaker",
    "male rapper",
    "female rapper",
    "beatboxing",
]  # do not consider multiple vocals.

path_list = glob.glob("/path/to/data/medleyDB/V1/*")


os.makedirs(
    f"/path/to/data/{args.samplerate_folder_name}/medleyDB_v1_rest",
    exist_ok=True,
)
os.makedirs(
    f"/path/to/data/{args.samplerate_folder_name}/medleyDB_v1_in_musdb",
    exist_ok=True,
)
for path in tqdm.tqdm(path_list):
    basename = os.path.basename(path)
    with open(
        f"/path/to/codes/medleydb/medleydb/data/Metadata/{basename}_METADATA.yaml"
    ) as stream:
        parsed_yaml = yaml.safe_load(stream)

    for key1, value1 in parsed_yaml["stems"].items():  # {'S01': {}, ...}
        for key2, value2 in value1["raw"].items():
            if value2["instrument"] in VOCALS:
                filename = value2["filename"]
                wav, sr = librosa.load(
                    f"{path}/{basename}_RAW/{filename}", sr=args.sample_rate, mono=True
                )
                intervals = librosa.effects.split(wav, top_db=65)
                for i in range(intervals.shape[0]):
                    filename_no_wav = filename.replace(".wav", "")
                    parsed_wav = wav[intervals[i, 0] : intervals[i, 1]]
                    if parsed_yaml["title"].replace(" ", "") in musdb_list:
                        sf.write(
                            f"/path/to/data/{args.samplerate_folder_name}/medleyDB_v1_in_musdb/{filename_no_wav}_{i}.wav",
                            parsed_wav,
                            args.sample_rate,
                            subtype="PCM_16",
                        )
                    else:
                        sf.write(
                            f"/path/to/data/{args.samplerate_folder_name}/medleyDB_v1_rest/{filename_no_wav}_{i}.wav",
                            parsed_wav,
                            args.sample_rate,
                            subtype="PCM_16",
                        )


path_list = glob.glob("/path/to/data/medleyDB/V2/*")

os.makedirs(
    f"/path/to/data/{args.samplerate_folder_name}/medleyDB_v2_rest",
    exist_ok=True,
)  # medleyDBv2 doesn't contain musdb dataset
for path in tqdm.tqdm(path_list):
    basename = os.path.basename(path)
    with open(
        f"/path/to/codes/medleydb/medleydb/data/Metadata/{basename}_METADATA.yaml"
    ) as stream:
        parsed_yaml = yaml.safe_load(stream)

    for key1, value1 in parsed_yaml["stems"].items():  # {'S01': {}, ...}
        for key2, value2 in value1["raw"].items():
            if value2["instrument"] in VOCALS:
                filename = value2["filename"]
                wav, sr = librosa.load(
                    f"{path}/{basename}_RAW/{filename}", sr=args.sample_rate, mono=True
                )
                intervals = librosa.effects.split(wav, top_db=65)
                for i in range(intervals.shape[0]):
                    filename_no_wav = filename.replace(".wav", "")
                    parsed_wav = wav[intervals[i, 0] : intervals[i, 1]]
                    sf.write(
                        f"/path/to/data/{args.samplerate_folder_name}/medleyDB_v2_rest/{filename_no_wav}_{i}.wav",
                        parsed_wav,
                        args.sample_rate,
                        subtype="PCM_16",
                    )
