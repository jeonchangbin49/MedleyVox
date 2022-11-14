import os
import math
import glob
import json
import librosa
import soundfile as sf
import tqdm
import sys

sys.path.append(os.path.abspath("../svs"))

from svs.utils import db2linear

save_dir = "/path/to/data/test_medleyDB"

sr = 44100

lengths = {"unison": 0, "duet": 0, "main_vs_rest": 0}


def save_wav_full(save_dir, data_root, json_root):
    json_paths = glob.glob(f"{json_root}/*.json")
    os.makedirs(save_dir, exist_ok=True)
    for json_path in tqdm.tqdm(json_paths):
        song_name = os.path.basename(json_path).replace(".json", "")
        with open(json_path, "r") as json_file:
            segments_info = json.load(json_file)

        name_wav_dict = {}
        song_name_list = []
        for key, item in segments_info.items():
            for gain_key, gain_item in item["gain_adjustment"].items():
                song_name_list.append(gain_key)
        song_name_list = list(set(song_name_list))
        print(song_name_list)
        for segment_name in song_name_list:
            name_wav_dict[segment_name] = librosa.load(
                f"{data_root}/{song_name}/{song_name}_RAW/{segment_name}.wav",
                sr=None,
                mono=True,
            )[0]

        for key, item in segments_info.items():
            for gain_key, gain_item in item["gain_adjustment"].items():
                name_wav_dict[gain_key][
                    librosa.time_to_samples(
                        item["start_sec"], sr=sr
                    ) : librosa.time_to_samples(item["end_sec"], sr=sr)
                ] = name_wav_dict[gain_key][
                    librosa.time_to_samples(
                        item["start_sec"], sr=sr
                    ) : librosa.time_to_samples(item["end_sec"], sr=sr)
                ] * db2linear(
                    item["gain_adjustment"][gain_key], eps=0.0
                )

        os.makedirs(f"{save_dir}/full/{song_name}", exist_ok=True)
        for segment_name in song_name_list:
            sf.write(
                f"{save_dir}/full/{song_name}/{segment_name}.wav",
                name_wav_dict[segment_name],
                sr,
            )


if __name__ == "__main__":
    v1_data_root = "/path/to/datamedleyDB/V1"
    v1_json_path = "./testset/testset_config/V1_rest_vocals_only_config"
    v2_data_root = "/path/to/datamedleyDB/V2"
    v2_json_path = "./testset/testset_config/V2_vocals_only_config"

    save_wav_full(save_dir, v1_data_root, v1_json_path)
    save_wav_full(save_dir, v2_data_root, v2_json_path)
