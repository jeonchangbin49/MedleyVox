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

train_list = glob.glob(
    "/path/to/data/177.k_multitimbre_guide_vocal/01.data/1.Training/original_data/*/*/*/*/*/*.wav"
)
valid_list = glob.glob(
    "/path/to/data/177.k_multitimbre_guide_vocal/01.data/2.Validation/original_data/*/*/*/*/*/*.wav"
)

path_list = train_list + valid_list


os.makedirs(
    f"/path/to/data/{args.samplerate_folder_name}/k_multitimbre",
    exist_ok=True,
)
for path in tqdm.tqdm(path_list):
    basename = os.path.basename(path).replace(".wav", "")
    wav, sr = librosa.load(path, sr=args.sample_rate, mono=True)
    sf.write(
        f"/path/to/data/{args.samplerate_folder_name}/k_multitimbre/{basename}.wav",
        wav,
        args.sample_rate,
        subtype="PCM_16",
    )
