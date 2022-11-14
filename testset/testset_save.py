import os
import glob
import json
import librosa
import soundfile as sf
import tqdm
import sys

sys.path.append(os.path.abspath("../svs"))

from svs.utils import load_wav_specific_position_mono, db2linear

save_dir = "/path/to/data/MedleyVox"

sr = 44100

lengths = {"unison": 0, "duet": 0, "main_vs_rest": 0}


def save_wav_segments(save_dir, data_root, json_root):
    json_paths = glob.glob(f"{json_root}/*.json")
    os.makedirs(save_dir, exist_ok=True)
    for json_path in tqdm.tqdm(json_paths):
        song_name = os.path.basename(json_path).replace(".json", "")
        with open(json_path, "r") as json_file:
            segments_info = json.load(json_file)
        for key, item in segments_info.items():
            seq_duration = item["end_sec"] - item["start_sec"]
            segment_type = item["type"]
            lengths[segment_type] = seq_duration + lengths[segment_type]
            segment_name = item["main_vocal"]
            wav = load_wav_specific_position_mono(
                f'{data_root}/{song_name}/{song_name}_RAW/{item["main_vocal"]}.wav',
                sr,
                seq_duration,
                item["start_sec"],
            )
            wav = wav * db2linear(item["gain_adjustment"][segment_name])
            os.makedirs(
                f"{save_dir}/{segment_type}/{song_name}/{key}/gt", exist_ok=True
            )
            sf.write(
                f"{save_dir}/{segment_type}/{song_name}/{key}/gt/{segment_name} - {key}.wav",
                wav,
                sr,
            )
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
                os.makedirs(
                    f"{save_dir}/{segment_type}/{song_name}/{key}/gt", exist_ok=True
                )
                sf.write(
                    f"{save_dir}/{segment_type}/{song_name}/{key}/gt/{other_vocal_name} - {key}.wav",
                    wav_other,
                    sr,
                )
                wav = wav + wav_other
            os.makedirs(
                f"{save_dir}/{segment_type}/{song_name}/{key}/mix", exist_ok=True
            )
            sf.write(
                f"{save_dir}/{segment_type}/{song_name}/{key}/mix/{song_name} - {key}.wav",
                wav,
                sr,
            )


if __name__ == "__main__":
    v1_data_root = "/path/to/data/medleyDB/V1"
    v1_json_path = "./testset/testset_config/V1_rest_vocals_only_config"
    v2_data_root = "/path/to/data/medleyDB/V2"
    v2_json_path = "./testset/testset_config/V2_vocals_only_config"

    save_wav_segments(save_dir, v1_data_root, v1_json_path)
    save_wav_segments(save_dir, v2_data_root, v2_json_path)

    print(lengths)
