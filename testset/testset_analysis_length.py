# to check the length of datasets in terms of # of singers, lengths, # of segments, # of songs.
import os
import glob
import json
import tqdm
import pandas as pd
import sys

sys.path.append(os.path.abspath("../svs"))


def compare(json_roots):
    df = pd.DataFrame(
        columns=[
            "seg_name",
            "song_name",
            "type",
            "num_of_singers",
            "num_of_voices",
            "start_sec",
            "end_sec",
            "length",
        ]
    )
    json_paths = []
    for json_root in json_roots:
        json_paths.extend(glob.glob(f"{json_root}/*.json"))
    for json_path in tqdm.tqdm(json_paths):
        song_name = os.path.basename(json_path).replace(".json", "")
        with open(json_path, "r") as json_file:
            segments_info = json.load(json_file)
        for key, item in segments_info.items():
            # First, check if data has a problem...
            assert item["song_name"] == song_name
            assert item["end_sec"] - item["start_sec"] > 0
            assert item["type"] in ["duet", "unison", "main_vs_rest"]
            if item["type"] == "chours":
                assert item["num_of_voices"] >= 3
            seq_duration = item["end_sec"] - item["start_sec"]
            del item["main_vocal"], item["other_vocals"], item["gain_adjustment"]
            item["length"] = seq_duration
            item["seg_name"] = f"{song_name} - {key}"
            df = pd.concat([df, pd.Series(item).to_frame().T], ignore_index=True)

    return df


if __name__ == "__main__":
    v1_json_path = "./testset/testset_config/V1_rest_vocals_only_config"
    v2_json_path = "./testset/testset_config/V2_vocals_only_config"

    a = compare([v1_json_path, v2_json_path])
    # a.groupby(['type','num_of_voices','num_of_singers'])['length'].sum()
    # a.groupby(['type','num_of_voices','num_of_singers'])['song_name'].count()
    # a.groupby(['type','num_of_voices','num_of_singers'])['song_name'].value_counts()
    # a.groupby(['type'])['song_name'].value_counts()
