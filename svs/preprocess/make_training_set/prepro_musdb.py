# To run this code, you first need to parse musdb-lyrics-(a), single vocal regions of musdb
# Check https://zenodo.org/record/3989267#.Y2JwxC_kHT8
# Schulze-Forster, K., Doire, C., Richard, G., & Badeau, R. "Phoneme Level Lyrics Alignment and Text-Informed Singing Voice Separation." IEEE/ACM Transactions on Audio, Speech and Language Processing (2021).

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
    "/path/to/codes/musdb_lyrics/single_regions_cut_24k/train/audio/vocals/a/*.wav"
)
test_list = glob.glob(
    "/path/to/codes/musdb_lyrics/single_regions_cut_24k/test/audio/vocals/a/*.wav"
)

os.makedirs(
    f"/path/to/data/{args.samplerate_folder_name}/musdb_a_train",
    exist_ok=True,
)
for path in tqdm.tqdm(train_list):
    basename = os.path.basename(path)
    wav, sr = librosa.load(path, sr=args.sample_rate, mono=True)
    sf.write(
        f"/path/to/data/{args.samplerate_folder_name}/musdb_a_train/{basename}",
        wav,
        args.sample_rate,
        subtype="PCM_16",
    )


os.makedirs(
    f"/path/to/data/{args.samplerate_folder_name}/musdb_a_test",
    exist_ok=True,
)
for path in tqdm.tqdm(test_list):
    basename = os.path.basename(path)
    wav, sr = librosa.load(path, sr=args.sample_rate, mono=True)
    sf.write(
        f"/path/to/data/{args.samplerate_folder_name}/musdb_a_test/{basename}",
        wav,
        args.sample_rate,
        subtype="PCM_16",
    )
