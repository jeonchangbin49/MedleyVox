import glob

import tqdm
import torchaudio


sr = 24000

data_list = [
    "/path/to/data/24k/CSD",
    "/path/to/data/24k/NUS",
    "/path/to/data/24k/TONAS",
    "/path/to/data/24k/VocalSet",
    "/path/to/data/24k/jsut-song_ver1",
    "/path/to/data/24k/jvs_music_ver1",
    "/path/to/data/24k/kiritan_revised",
    "/path/to/data/24k/vocadito",
    "/path/to/data/24k/musdb_a_train",
    "/path/to/data/24k/OpenSinger",
    "/path/to/data/24k/medleyDB_v1_in_musdb",
    # '/path/to/data/24k/medleyDB_v1_rest',
    # '/path/to/data/24k/medleyDB_v2_rest',
    "/path/to/data/24k/k_multisinger",
    "/path/to/data/24k/k_multitimbre",
]


lengths = []
for data_root in tqdm.tqdm(data_list):
    song_list = glob.glob(f"{data_root}/*.wav")
    for song in song_list:
        length = torchaudio.info(song).num_frames
        lengths.append(length / sr)  # in [sec]

print(f"total length {sum(lengths) / 60} mins")
