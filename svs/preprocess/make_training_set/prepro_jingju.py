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

path_list = glob.glob("/path/to/data/jingju/*/*/*.wav")

os.makedirs(
    f"/path/to/data/{args.samplerate_folder_name}/jingju",
    exist_ok=True,
)
print(len(path_list))
for path in tqdm.tqdm(path_list):
    basename = os.path.basename(path)
    wav, sr = librosa.load(path, sr=args.sample_rate, mono=True)
    sf.write(
        f"/path/to/data/{args.samplerate_folder_name}/jingju/{basename}",
        wav,
        args.sample_rate,
        subtype="PCM_16",
    )
