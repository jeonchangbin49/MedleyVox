import os
import glob
import argparse

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


path_1 = glob.glob(
    "/path/to/data/OpenSinger/ManRaw/*/*.wav"
)
path_2 = glob.glob(
    "/path/to/data/OpenSinger/WomanRaw/*/*.wav"
)

path_list = path_1 + path_2

os.makedirs(
    f"/path/to/data/{args.samplerate_folder_name}/OpenSinger",
    exist_ok=True,
)

for path in tqdm.tqdm(path_list):
    basename = os.path.basename(path)
    path_split = path.split("/")
    wav, sr = librosa.load(path, sr=args.sample_rate, mono=True)
    sf.write(
        f"/path/to/data/{args.samplerate_folder_name}/OpenSinger/{path_split[-3]}_{basename}",
        wav,
        args.sample_rate,
        subtype="PCM_16",
    )
