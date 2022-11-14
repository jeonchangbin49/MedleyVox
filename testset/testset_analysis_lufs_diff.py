# to check the rough loudness difference between main vocals and their corresponding others
import os
import glob
import json
import librosa
import soundfile as sf
import tqdm
import sys

sys.path.append(os.path.abspath("../svs"))

from svs.utils import load_wav_specific_position_mono, db2linear, linear2db

import pyloudnorm as pyln

sr = 44100
meter = pyln.Meter(sr)

# lengths = {'unison':0, 'duet':0, 'main_vs_rest':0}


def compare(data_root, json_root):
    list_lufs_diff = {}
    json_paths = glob.glob(f"{json_root}/*.json")
    for json_path in tqdm.tqdm(json_paths):
        song_name = os.path.basename(json_path).replace(".json", "")
        with open(json_path, "r") as json_file:
            segments_info = json.load(json_file)
        for key, item in segments_info.items():
            seq_duration = item["end_sec"] - item["start_sec"]
            segment_type = item["type"]
            if segment_type == "main_vs_rest":
                # lengths[segment_type] = seq_duration + lengths[segment_type]
                segment_name = item["main_vocal"]
                wav = load_wav_specific_position_mono(
                    f'{data_root}/{song_name}/{song_name}_RAW/{item["main_vocal"]}.wav',
                    sr,
                    seq_duration,
                    item["start_sec"],
                )
                wav = wav * db2linear(item["gain_adjustment"][segment_name])
                lufs_1 = meter.integrated_loudness(wav)
                wav_other = 0
                for other_vocal_name in item["other_vocals"]:
                    wav_other = load_wav_specific_position_mono(
                        f"{data_root}/{song_name}/{song_name}_RAW/{other_vocal_name}.wav",
                        sr,
                        seq_duration,
                        item["start_sec"],
                    )
                    wav_other = wav_other * db2linear(
                        item["gain_adjustment"][other_vocal_name]
                    )
                lufs_2 = meter.integrated_loudness(wav_other)
                list_lufs_diff[f"{song_name}_{key}"] = lufs_1 - lufs_2
    return list_lufs_diff


v1_data_root = "/path/to/datamedleyDB/V1"
v1_json_path = "./testset/testset_config/V1_rest_vocals_only_config"
v2_data_root = "/path/to/datamedleyDB/V2"
v2_json_path = "./testset/testset_config/V2_vocals_only_config"

list_lufs_diff_1 = compare(v1_data_root, v1_json_path)
list_lufs_diff_2 = compare(v2_data_root, v2_json_path)
