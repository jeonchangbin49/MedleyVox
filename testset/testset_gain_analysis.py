# to check the gain_adjustment parameters of every segment
import os
import glob
import json
from pprint import pprint


def gain_compare(json_path):
    song_name = os.path.basename(json_path).replace(".json", "")
    with open(json_path, "r") as json_file:
        segments_info = json.load(json_file)
    name_gain_dict = {}
    song_name_list = []
    for key, item in segments_info.items():
        for gain_key, gain_item in item["gain_adjustment"].items():
            song_name_list.append(gain_key)
    song_name_list = list(set(song_name_list))
    for segment_name in song_name_list:
        name_gain_dict[segment_name] = []
    for key, item in segments_info.items():
        for gain_key, gain_item in item["gain_adjustment"].items():
            name_gain_dict[gain_key].append(gain_item)

    print(song_name)
    pprint(name_gain_dict)


if __name__ == "__main__":
    v1_json_path = "./testset/testset_config/V1_rest_vocals_only_config"
    v2_json_path = "./testset/testset_config/V2_vocals_only_config"

    v1_path_list = glob.glob(f"{v1_json_path}/*.json")

    for v1_path in v1_path_list:
        gain_compare(v1_path)

    # v2_path_list = glob.glob(f'{v2_json_path}/*.json')

    # for v2_path in v2_path_list:
    #     gain_compare(v2_path)
