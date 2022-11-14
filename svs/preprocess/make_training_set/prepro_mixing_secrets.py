# due to the behavior of 'librosa.effects.split', you might need to manually remove the small size files that created from this code.

import os
import glob

import librosa
import soundfile as sf
import tqdm


VOCALS = [
    "Vocals",
    "choir",
    "Choir",
    "LeadVox",
    "LeadVoxHi",
    "LeadVoxLo",
    "LeadVoxDoubletrack",
    "LeadVoxChorus",
    "ElectroChoir",
    "LeadVoxTapeEffect",
    "LeadVoxRaw",
    "VerseVox",
    "ChorusVox",
    "LeadVoxAlt",
    "LeadVoxAltDT",
    "BackingVox",
    "LeadVoxDTs",
    "LeadVoxDT",
    "Vox",
    "VoxMid",
    "SynthChoir",
    "VoxAdLibs",
    "VoxAdLib",
    "VoxMidAlt",
    "SampleChoir",
    "LeadVoxDuet",
    "LeadVoxDoubles",
    "Speech",
    "VoxSFX",
    "LeadVoxSFX",
    "BackingVoxSFX",
    "LeadVoxScratch",
    "VoxFX",
    "LeadVoxFX",
    "VocalSFX",
    "VOX",
    "LeadVoxMic",
    "LeadVoxDTMic",
    "BackingVos",
    "Vocoder",
    "LeadVoxHarm",
    "LeadVoxRoom",
    "BackingVoxDT",
    "VocoderFX",
    "VoxSample",
    "BackingVoxFemale",
    "BackingVoxMaleSubmix",
    "BackingVoxHarmonizer",
    "LeadVoxDTAdLib",
    "Leadvox",
    "LeadVoxDouble",
    "LeadVoxCh",
    "LeadVoxChDT",
    "BackingVoxCh",
    "BackingVoxBr",
    "LeadVoxVs",
    "SFXLaugh",
    "Laugh",
    "Scream",
]

path_list = glob.glob(
    "/path/to/data/mixing_secrets_in_musdb_train/*/*/*.wav"
)


os.makedirs(
    "/path/to/data/24k/mixing_secrets_in_musdb_train",
    exist_ok=True,
)
for path in tqdm.tqdm(path_list):
    basename = os.path.basename(path).replace(".wav", "")
    song_name = path.split("/")[-2].replace("_Full", "")
    track_category = basename.split("_")[1]  # LeadVox3
    for category in VOCALS:
        if track_category.count(category) >= 1:
            wav, sr = librosa.load(path, sr=16000, mono=True)
            intervals = librosa.effects.split(wav, top_db=65)
            for i in range(intervals.shape[0]):
                parsed_wav = wav[intervals[i, 0] : intervals[i, 1]]
                sf.write(
                    f"/path/to/data/24k/mixing_secrets_in_musdb_train/{song_name}_{track_category}_{i}.wav",
                    parsed_wav,
                    16000,
                    subtype="PCM_16",
                )
            break
        else:
            continue


path_list = glob.glob(
    "/path/to/data/mixing_secrets_rest_musdb18_train/*/*/*.wav"
)


os.makedirs(
    "/path/to/data/24k/mixing_secrets_rest", exist_ok=True
)
for path in tqdm.tqdm(path_list):
    basename = os.path.basename(path).replace(".wav", "")
    song_name = path.split("/")[-2].replace("_Full", "")
    track_category = basename.split("_")[-1]  # LeadVox3
    for category in VOCALS:
        if track_category.count(category) >= 1:
            wav, sr = librosa.load(path, sr=16000, mono=True)
            intervals = librosa.effects.split(wav, top_db=65)
            for i in range(intervals.shape[0]):
                parsed_wav = wav[intervals[i, 0] : intervals[i, 1]]
                sf.write(
                    f"/path/to/data/24k/mixing_secrets_rest/{song_name}_{track_category}_{i}.wav",
                    parsed_wav,
                    16000,
                    subtype="PCM_16",
                )
            break
        else:
            continue
