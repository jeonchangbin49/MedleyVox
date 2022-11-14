import os
import math
import json
import random
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
import pyloudnorm as pyln

from ..utils import (
    load_wav_arbitrary_position_mono,
    load_wav_specific_position_mono,
    db2linear,
    loudness_match_and_norm,
    loudness_normal_match_and_norm,
    loudnorm,
    change_pitch_and_formant_random,
    worker_init_fn,
    change_pitch_and_formant,
)


# Singing dataset + LibriSpeech dataset
class DuetSingingSpeechMixTraining(Dataset):
    """Dataset class for duet singing voice separation tasks.

    Args:
        singing_data_dir (List): The paths of the directories of singing data.
        speech_data_dir (List) : The paths of the directories of speech data.
        song_length_dict_path (str) : The path that contains the length information of singing train data.
        same_song_dict_path (List of list) : The list of lists. Each list is made of
            [
                'root path of singing data',
                'path of json file that contains the dictionary of same song's list'
                'dataset name'
            ]
        same_singer_dict_path (List of list) : The list of lists. Each list is made of
            [
                'root path of singing data',
                'path of json file that contains the dictionary of same singer's songs'
                'dataset name'
            ]
        same_speaker_dict_path (List of list) : The list of lists. Each list is made of
            [
                'root path of speech data',
                'path of json file that contains the dictionary of same speaker's speeches',
                'dataset name
            ]
        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture.
        segment (float) : The desired sources and mixtures length in [second].
        unison_prob (float) : Probability of applying unison data augmentation. 0 <= unison_prob <=1
        pitch_formant_augment_prob (float) : Probability of applying pitch and formant shift augmentation. 0 <= prob <=1
        augment (bool) : If true, the volume of two input sources are roughly matched and the loudness of mixture is normalized to -24 LUFS
        part_of_data (float) : Use reduced amount of training data. If part_of_data == 0.1, only 10% of training data will be used.
        sing_sing_ratio (float) : Case 1. Ratio of 'different singing + singing' in training data sampling process.
        sing_speech_ratio (float) : Case 2. Ratio of 'different singing + speech' in training data sampling process.
        same_song_ratio (float) : Case 3. Ratio of 'same song of different singers’ in training data sampling process.
        same_singer_ratio (float) : Case 4. Ratio of 'different songs of same singer’ in training data sampling process.
        same_speaker_ratio (float) : Case 5. Ratio of 'different speeches of same speaker’ in training data sampling process.
        speech_speech_ratio (not specified) : Case 6. Ratio of 'different speech + speech’ in training data sampling process. 
                                            This is not specified by arguments, but automatically calculated by ‘1 - (sum_of_rest_arguments)’.
    Notes:
        sum_of_ratios = (sing_sing_ratio
            + sing_speech_ratio
            + same_song_ratio
            + same_singer_ratio
            + same_speaker_ratio)
        should be smaller than 1
        speech_speech_ratio (different speech + speech, Case 6) will be automatically calculated as 1 - sum_of_ratios
    """

    dataset_name = "singing_with_speech"

    def __init__(
        self,
        singing_data_dir,
        speech_data_dir,
        song_length_dict_path,
        same_song_dict_path,
        same_singer_dict_path,
        same_speaker_dict_path,
        sample_rate=16000,
        n_src=2,
        segment=3,
        unison_prob=0.3,
        pitch_formant_augment_prob=0.3,
        augment=True,
        part_of_data=None,
        reduced_training_data_ratio=1.0,
        sing_sing_ratio=0.2,
        sing_speech_ratio=0.2,
        same_song_ratio=0.15,
        same_singer_ratio=0.15,
        same_speaker_ratio=0.15,
        # speech_speech_ratio=0.15
    ):
        self.segment = segment  # segment is length of input segment
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.augment = augment
        self.unison_prob = unison_prob
        self.pitch_formant_augment_prob = pitch_formant_augment_prob
        self.meter = pyln.Meter(self.sample_rate)
        self.reduced_training_data_ratio = reduced_training_data_ratio

        # load singing_data_list from the list of singing data dirs
        self.singing_wav_paths = []
        for path in singing_data_dir:
            self.singing_wav_paths.extend(glob.glob(f"{path}/*.wav"))

        # load speech_data_list from the list of speech data dirs
        self.speech_wav_paths = []
        for path in speech_data_dir:
            self.speech_wav_paths.extend(glob.glob(f"{path}/*.wav"))

        # to check the influence of the training data size, reduce the number of training data.
        if part_of_data != None:
            print("before number of singing data  :", len(self.singing_wav_paths))
            self.singing_wav_paths = random.sample(
                self.singing_wav_paths, int(len(self.singing_wav_paths) * part_of_data)
            )
            print("after number of singing data :", len(self.singing_wav_paths))

            print("before number of speech data  :", len(self.speech_wav_paths))
            self.speech_wav_paths = random.sample(
                self.speech_wav_paths, int(len(self.speech_wav_paths) * part_of_data)
            )
            print("after number of speech data :", len(self.speech_wav_paths))

        song_name_path_dict = {}
        for data_path in self.singing_wav_paths:
            song_name_path_dict[os.path.basename(data_path)] = data_path

        # We have to load a long song more than a short song.
        # Therefore, we will make a singing_train_list, which contains training data paths,
        # a long song more often than a short song.
        with open(song_length_dict_path, "r") as json_file:
            song_length_dict = json.load(json_file)
        # sort dict by descending order
        song_length_dict = dict(
            sorted(song_length_dict.items(), key=lambda x: x[1], reverse=True)
        )
        song_names = []
        song_lengths = []
        for key, value in song_length_dict.items():
            if key not in song_name_path_dict:
                pass
            else:
                song_names.append(key)
                song_lengths.append(value)

        # Determine how many times to load one audio file during one epoch
        train_list_number = np.array(song_lengths) / (self.segment * self.sample_rate)
        self.singing_train_list = []
        for i, num_seg in enumerate(list(train_list_number)):
            try:
                self.singing_train_list.extend(
                    [song_name_path_dict[song_names[i]]] * math.ceil(num_seg)
                )
            except KeyError:  # some songs might not be in the self.song_name_path_dict
                pass

        self.len_singing_train_list = len(self.singing_train_list)

        self.sing_sing_ratio_cum = sing_sing_ratio
        self.sing_speech_ratio_cum = self.sing_sing_ratio_cum + sing_speech_ratio
        self.same_song_ratio_cum = self.sing_speech_ratio_cum + same_song_ratio
        self.same_singer_ratio_cum = self.same_song_ratio_cum + same_singer_ratio
        self.same_speaker_ratio_cum = self.same_singer_ratio_cum + same_speaker_ratio

        self.same_song_dict = {}  # {'songname':[...], ...}
        self.same_song_dataname_path_dict = (
            {}
        )  # {'OpenSinger':OpenSinger root path, ...}
        self.same_song_list = (
            []
        )  # [{'filename':filename,'dataset':dataname,'songname':songname}, ...]
        if same_song_dict_path != None:
            # same_song_dict_path [[data_root,data_dict_path, data_name], ...]
            for path in same_song_dict_path:
                with open(path[1], "r") as json_file:
                    same_song_dict_temp = json.load(json_file)
                self.same_song_dict.update(same_song_dict_temp)
                self.same_song_dataname_path_dict[path[2]] = path[0]
                for same_song_key, same_song_value in same_song_dict_temp.items():
                    for same_song_value_item in same_song_value:
                        self.same_song_list.append(
                            {
                                "filename": same_song_value_item,  # in case of 'same song', key 'filename' contains item ['filename', 'audio_length']
                                "dataset": path[2],
                                "songname": same_song_key,
                            }
                        )

        self.same_singer_dict = {}  # {'singername':[...], ...}
        self.same_singer_dataname_path_dict = (
            {}
        )  # {'OpenSinger':OpenSinger root path, ...}
        self.same_singer_list = (
            []
        )  # [{'filename':filename,'dataset':dataname,'songname':songname}, ...]
        if same_singer_dict_path != None:
            # same_singer_dict_path [[data_root,data_dict_path, data_name]]
            for path in same_singer_dict_path:
                with open(path[1], "r") as json_file:
                    same_singer_dict_temp = json.load(json_file)
                self.same_singer_dict.update(same_singer_dict_temp)
                self.same_singer_dataname_path_dict[path[2]] = path[0]
                for same_singer_key, same_singer_value in same_singer_dict_temp.items():
                    for same_singer_value_item in same_singer_value:
                        self.same_singer_list.append(
                            {
                                "filename": same_singer_value_item,
                                "dataset": path[2],
                                "singername": same_singer_key,
                            }
                        )

        self.same_speaker_dict = {}  # {'speakername':[...], ...}
        self.same_speaker_dataname_path_dict = (
            {}
        )  # {'LibriSpeech_train-clean-100':LibriSpeech_train-clean-100 root path, ...}
        self.same_speaker_list = (
            []
        )  # [{'filename':filename,'dataset':dataname,'songname':songname}, ...]
        if same_speaker_dict_path != None:
            # same_speaker_dict_path [[data_root,data_dict_path, data_name]]
            for path in same_speaker_dict_path:
                with open(path[1], "r") as json_file:
                    same_speaker_dict_temp = json.load(json_file)
                self.same_speaker_dict.update(same_speaker_dict_temp)
                self.same_speaker_dataname_path_dict[path[2]] = path[0]
                for (
                    same_speaker_key,
                    same_speaker_value,
                ) in same_speaker_dict_temp.items():
                    for same_speaker_value_item in same_speaker_value:
                        self.same_speaker_list.append(
                            {
                                "filename": same_speaker_value_item,
                                "dataset": path[2],
                                "speakername": same_speaker_key,
                            }
                        )

    def __len__(self):
        return int(self.len_singing_train_list * self.reduced_training_data_ratio)

    def return_paths(self, idx, rand_prob):
        # Case 1. different singing + singing
        if rand_prob <= self.sing_sing_ratio_cum:
            data_path_1 = self.singing_train_list[idx]
            data_path_2 = random.choice(self.singing_train_list)
            return {"path_1": data_path_1, "path_2": data_path_2}

        # Case 2. singing + speech
        elif self.sing_sing_ratio_cum < rand_prob <= self.sing_speech_ratio_cum:
            data_path_1 = self.singing_train_list[idx]
            data_path_2 = random.choice(self.speech_wav_paths)
            return {"path_1": data_path_1, "path_2": data_path_2}

        # Case 3. same song (but with different singer)
        elif self.sing_speech_ratio_cum < rand_prob <= self.same_song_ratio_cum:
            song_dict = random.choice(
                self.same_song_list
            )  # First, randomly choose a song (song_1)
            data_root = self.same_song_dataname_path_dict[
                song_dict["dataset"]
            ]  # Return the name of song_1's dataset
            filename_1 = song_dict[
                "filename"
            ]  # filename_1 => ['filename', 'audio_length']. Return the list that contains the filename and audio_length [sec] of song_1.
            songname_1 = song_dict[
                "songname"
            ]  # Return the song name (not the path or basename of file) of song_1
            data_path_1 = (
                f"{data_root}/{filename_1[0]}.wav"  # Return the data path of song_1
            )
            same_song_1_list = self.same_song_dict[
                songname_1
            ].copy()  # Copy the list of song_1's other speeches.
            # For same songs, we need to store the audio lengths because we are going to sample audio segments from the exact same positions
            audio_length_1 = filename_1[1]  # Return the audio length [sec] of song_1
            same_song_1_list.remove(
                filename_1
            )  # Before randomly choose song_2, remove song_1 from the song_1's other speeches.
            filename_2 = random.choice(
                same_song_1_list
            )  # Randomly choose a speech_2 from same_song_1_list
            data_path_2 = f"{data_root}/{filename_2[0]}.wav"  # Return the data path of data_path_2
            audio_length_2 = filename_2[1]  # Return the audio length [sec] of song_2
            same_song_1_list.remove(
                filename_2
            )  # Remove metadata of sampled files for future usage in multi_singing_libri_dataset.py
            return {
                "path_1": data_path_1,
                "path_2": data_path_2,
                "same_list": same_song_1_list,
                "audio_len_1": audio_length_1,
                "audio_len_2": audio_length_2,
                "data_root": data_root,
            }

        # Case 4. same singer (but with different song)
        elif self.same_song_ratio_cum < rand_prob <= self.same_singer_ratio_cum:
            singer_dict = random.choice(
                self.same_singer_list
            )  # First, randomly choose a song (song_1)
            data_root = self.same_singer_dataname_path_dict[
                singer_dict["dataset"]
            ]  # Return the name of song_1's dataset
            filename_1 = singer_dict[
                "filename"
            ]  # Return the name of the filename of song_1
            singername_1 = singer_dict[
                "singername"
            ]  # Return the name of the singer name of song_1
            data_path_1 = (
                f"{data_root}/{filename_1}.wav"  # Return the data path of song_1
            )
            same_singer_1_list = self.same_singer_dict[
                singername_1
            ].copy()  # Copy the list of singer_1's other songs.
            same_singer_1_list.remove(
                filename_1
            )  # Before randomly choose another song of singer_1, remove filename_1 from the list first.
            filename_2 = random.choice(same_singer_1_list)  # Randomly choose a song_2
            data_path_2 = (
                f"{data_root}/{filename_2}.wav"  # Return the data path of song_2
            )
            same_singer_1_list.remove(
                filename_2
            )  # Remove metadata of sampled files for future usage in multi_singing_libri_dataset.py
            return {
                "path_1": data_path_1,
                "path_2": data_path_2,
                "same_list": same_singer_1_list,
                "data_root": data_root,
            }

        # Case 5. same speaker (but with different speech)
        elif self.same_singer_ratio_cum < rand_prob <= self.same_speaker_ratio_cum:
            speaker_dict = random.choice(
                self.same_speaker_list
            )  # First, randomly choose a speech (speech_1)
            data_root = self.same_speaker_dataname_path_dict[
                speaker_dict["dataset"]
            ]  # Return the name of speech_1's dataset
            filename_1 = speaker_dict[
                "filename"
            ]  # Return the name of the filename of speech_1
            speakername_1 = speaker_dict[
                "speakername"
            ]  # Return the speaker name of speech_1
            data_path_1 = (
                f"{data_root}/{filename_1}.wav"  # Return the data path of speech_1
            )
            same_speaker_1_list = self.same_speaker_dict[
                speakername_1
            ].copy()  # Copy the list of speaker_1's other speeches.
            same_speaker_1_list.remove(
                filename_1
            )  # Before randomly choose another speech of speaker_1, remove speech_1 from the list first.
            filename_2 = random.choice(
                same_speaker_1_list
            )  # Randomly choose a speech_2
            data_path_2 = (
                f"{data_root}/{filename_2}.wav"  # Return the data path of speech_2
            )
            same_speaker_1_list.remove(
                filename_2
            )  # Remove metadata of sampled files for future usage in multi_singing_libri_dataset.py
            return {
                "path_1": data_path_1,
                "path_2": data_path_2,
                "same_list": same_speaker_1_list,
                "data_root": data_root,
            }

        # Case 6. different speech + speech
        elif self.same_speaker_ratio_cum < rand_prob <= 1:
            data_path_1 = random.choice(self.speech_wav_paths)
            data_path_2 = random.choice(self.speech_wav_paths)
            return {"path_1": data_path_1, "path_2": data_path_2}

    def load_first_audio(self, rand_prob, paths_info_dict):
        # Case 3, same song (but with different singer)
        if self.sing_speech_ratio_cum < rand_prob <= self.same_song_ratio_cum:
            min_audio_length = min(
                paths_info_dict["audio_len_1"], paths_info_dict["audio_len_2"]
            )  # Choose smaller audio_length of two chosen data
            rand_start = random.uniform(
                0, min_audio_length - self.segment
            )  # Randomly choose the start position of audio
            source_1 = load_wav_specific_position_mono(
                paths_info_dict["path_1"], self.sample_rate, self.segment, rand_start
            )  # if rand_start is smaller than 0, starting position will be converted to zero.
            paths_info_dict["rand_start"] = rand_start
            return source_1, paths_info_dict
        # Other Cases
        else:
            source_1 = load_wav_arbitrary_position_mono(
                paths_info_dict["path_1"], self.sample_rate, self.segment
            )
            return source_1, paths_info_dict

    def load_second_audio(self, source_1, rand_prob, augment_prob, paths_info_dict):
        # unison augmentation
        # In Case 1 (diff sing+sing) or Case 6 (diff speech+speech), apply unison augmentation. data_path_2 will be neglected.
        if (augment_prob <= self.unison_prob) and (
            rand_prob <= self.sing_sing_ratio_cum
            or self.same_speaker_ratio_cum < rand_prob <= 1
        ):
            source_2 = change_pitch_and_formant_random(source_1, self.sample_rate)

        # pitch+formant augmentation only on one source
        elif (
            self.unison_prob
            < augment_prob
            <= self.unison_prob + self.pitch_formant_augment_prob
        ):
            # pitch+formant augment on Case 3, same song (but with different singer)
            if self.sing_speech_ratio_cum < rand_prob <= self.same_song_ratio_cum:
                source_2 = load_wav_specific_position_mono(
                    paths_info_dict["path_2"],
                    self.sample_rate,
                    self.segment,
                    paths_info_dict["rand_start"],
                )
            # pitch+formant augment on other Cases.
            else:
                source_2 = load_wav_arbitrary_position_mono(
                    paths_info_dict["path_2"], self.sample_rate, self.segment
                )
            which_source_prob = random.random()
            if (
                which_source_prob <= 0.333
            ):  # apply pitch+formant augmentation to source_1 or source_2
                source_2 = change_pitch_and_formant_random(source_2, self.sample_rate)
            elif 0.333 < which_source_prob <= 0.666:
                source_1 = change_pitch_and_formant_random(source_1, self.sample_rate)
            else:
                source_1 = change_pitch_and_formant_random(source_1, self.sample_rate)
                source_2 = change_pitch_and_formant_random(source_2, self.sample_rate)
        # No augmentation. Case 3, same song (but with different singer)
        elif self.sing_speech_ratio_cum < rand_prob <= self.same_song_ratio_cum:
            source_2 = load_wav_specific_position_mono(
                paths_info_dict["path_2"],
                self.sample_rate,
                self.segment,
                paths_info_dict["rand_start"],
            )
        # No augmentation. Other Cases.
        else:
            source_2 = load_wav_arbitrary_position_mono(
                paths_info_dict["path_2"], self.sample_rate, self.segment
            )

        return source_1, source_2

    def __getitem__(self, idx):
        # Load two audio paths first.
        rand_prob = random.random()

        paths_info_dict = self.return_paths(idx, rand_prob)

        # Load audio
        sources_list = []

        source_1, paths_info_dict = self.load_first_audio(rand_prob, paths_info_dict)

        augment_prob = random.random()
        source_1, source_2 = self.load_second_audio(
            source_1, rand_prob, augment_prob, paths_info_dict
        )

        # Apply loudness normalization and math between source_1 and source_2
        if self.augment:
            source_1, source_2 = loudness_normal_match_and_norm(
                source_1, source_2, self.meter
            )

        mixture = source_1 + source_2
        mixture, adjusted_gain = loudnorm(
            mixture, -24.0, self.meter
        )  # -24 is target_lufs. -14 is too hot.
        source_1 = source_1 * db2linear(adjusted_gain)
        source_2 = source_2 * db2linear(adjusted_gain)

        sources_list.append(source_1)
        sources_list.append(source_2)

        # Convert to torch tensor
        mixture = torch.as_tensor(mixture, dtype=torch.float32)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.as_tensor(sources, dtype=torch.float32)

        return mixture, sources


class DuetSingingSpeechMixValidation(Dataset):
    """Dataset class for duet source separation. For validation dataset
    Args:
        data_dir (List) : The list of lists. Each list is made of
            [
                'root path of data for source_1',
                'root path of data for source_2',
                'path of json file that contains metadata of sources',
                'data name (ex. sing_sing_diff)'
            ]
        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture.
        segment (int, optional) : The desired sources and mixtures length in s.
        augment (bool) : If true, the volume of two input sources are roughly matched and the loudness of mixture is normalized to -24 LUFS
    """

    dataset_name = "singing_with_speech_valid"

    def __init__(
        self,
        data_dir,
        sample_rate=16000,
        n_src=2,
        segment=3,
        augment=True,
    ):
        self.source_1_paths = []
        self.source_2_paths = []
        self.metadata_list = []

        for data_dir_set in data_dir:

            with open(data_dir_set[2], "r") as json_file:
                self.valid_regions_dict = json.load(json_file)
            for key, value in self.valid_regions_dict.items():
                self.source_1_paths.append(f"{data_dir_set[0]}/{key}")
                self.source_2_paths.append(
                    f'{data_dir_set[1]}/{value["corresponding_data"]}'
                )
                self.metadata_list.append(value)

        self.segment = segment
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.augment = augment
        self.meter = pyln.Meter(self.sample_rate)

    def __len__(self):
        return len(self.source_1_paths)

    def __getitem__(self, idx):
        data_path_1 = self.source_1_paths[idx]
        data_path_2 = self.source_2_paths[idx]
        metadata = self.metadata_list[idx]

        sources_list = []

        source_1 = load_wav_specific_position_mono(
            data_path_1, self.sample_rate, self.segment, 0.0
        )  # data_1 starts from 0. sec
        source_2 = load_wav_specific_position_mono(
            data_path_2, self.sample_rate, self.segment, metadata["position(sec)"]
        )

        if metadata["unison_aug"]:
            source_2 = change_pitch_and_formant(
                source_2,
                self.sample_rate,
                metadata["unison_params"][0],
                metadata["unison_params"][1],
                1,
                metadata["unison_params"][3],
            )

        if self.augment:
            source_1, source_2 = loudness_match_and_norm(source_1, source_2, self.meter)

        mixture = source_1 + source_2
        mixture, adjusted_gain = loudnorm(
            mixture, -24.0, self.meter
        )  # -24 is target_lufs. -14 is too hot.
        source_1 = source_1 * db2linear(adjusted_gain)
        source_2 = source_2 * db2linear(adjusted_gain)

        sources_list.append(source_1)
        sources_list.append(source_2)

        # Convert to torch tensor
        mixture = torch.as_tensor(mixture, dtype=torch.float32)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.as_tensor(sources, dtype=torch.float32)

        return mixture, sources


# check the loader
if __name__ == "__main__":

    import argparse
    import tqdm

    parser = argparse.ArgumentParser(description="Singing separation dataset checker")

    # Dataset paramaters
    parser.add_argument(
        "--dataset",
        type=str,
        default="singing_librispeech",
        choices=["librimix", "singing_sep", "singing_librispeech"],
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--sing_sing_ratio",
        type=float,
        default=0.2,
        help="singing+singing train dataset portion",
    )
    parser.add_argument(
        "--sing_speech_ratio",
        type=float,
        default=0.2,
        help="singing+singing train dataset portion",
    )
    parser.add_argument(
        "--same_song_ratio",
        type=float,
        default=0.15,
        help="singing+singing train dataset portion",
    )
    parser.add_argument(
        "--same_singer_ratio",
        type=float,
        default=0.15,
        help="singing+singing train dataset portion",
    )
    parser.add_argument(
        "--same_speaker_ratio",
        type=float,
        default=0.15,
        help="singing+singing train dataset portion",
    )

    parser.add_argument(
        "--train_root",
        nargs="+",
        default=[
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
            "/path/to/data/24k/medleyDB_v1_rest",
            "/path/to/data/24k/medleyDB_v2_rest",
            "/path/to/data/24k/k_multisinger",
            "/path/to/data/24k/k_multitimbre",
        ],
        help="root path list of dataset",
    )
    parser.add_argument(
        "--speech_train_root",
        nargs="+",
        default=[
            "/path/to/data/24k/LibriSpeech_train-clean-360",
            "/path/to/data/24k/LibriSpeech_train-clean-100",
        ],
        help="root path list of dataset",
    )
    parser.add_argument(
        "--same_song_dict_path",
        nargs="+",
        # type=str,
        action="append",
        default=[
            # ['/path/to/data/24k/OpenSinger', './svs/preprocess/make_same_song_dict/same_song_dict_OpenSinger.json',  'OpenSinger'],
            [
                "/path/to/data/24k/k_multisinger",
                "./svs/preprocess/make_same_song_dict/same_song_k_multisinger.json",
                "k_multisinger",
            ]
        ],
        help="For making the dataloader that outputs source1 and source2 from SAME song. list of [[data_root,data_dict_path, data_name], ...]",
    )
    parser.add_argument(
        "--same_singer_dict_path",
        nargs="+",
        # type=str,
        action="append",
        default=[
            [
                "/path/to/data/24k/OpenSinger",
                "./svs/preprocess/make_same_singer_dict/same_singer_OpenSinger.json",
                "OpenSinger",
            ],
            [
                "/path/to/data/24k/k_multisinger",
                "./svs/preprocess/make_same_singer_dict/same_singer_k_multisinger.json",
                "k_multisinger",
            ],
            [
                "/path/to/data/24k/CSD",
                "./svs/preprocess/make_same_singer_dict/same_singer_CSD.json",
                "CSD",
            ],
            [
                "/path/to/data/24k/jsut-song_ver1",
                "./svs/preprocess/make_same_singer_dict/same_singer_jsut-song_ver1.json",
                "jsut-song_ver1",
            ],
            [
                "/path/to/data/24k/jvs_music_ver1",
                "./svs/preprocess/make_same_singer_dict/same_singer_jvs_music_ver1.json",
                "jvs_music_ver1",
            ],
            [
                "/path/to/data/24k/k_multitimbre",
                "./svs/preprocess/make_same_singer_dict/same_singer_k_multitimbre.json",
                "k_multitimbre",
            ],
            [
                "/path/to/data/24k/kiritan_revised",
                "./svs/preprocess/make_same_singer_dict/same_singer_kiritan.json",
                "kiritan",
            ],
            [
                "/path/to/data/24k/musdb_a_train",
                "./svs/preprocess/make_same_singer_dict/same_singer_musdb_a_train.json",
                "musdb_a_train",
            ],
            [
                "/path/to/data/24k/NUS",
                "./svs/preprocess/make_same_singer_dict/same_singer_NUS.json",
                "NUS",
            ],
            [
                "/path/to/data/24k/VocalSet",
                "./svs/preprocess/make_same_singer_dict/same_singer_VocalSet.json",
                "VocalSet",
            ],
        ],
        help="For making the dataloader that outputs source1 and source2 from SAME singer. list of [[data_root,data_dict_path, data_name], ...]",
    )
    parser.add_argument(
        "--same_speaker_dict_path",
        nargs="+",
        # type=str,
        action="append",
        default=[
            [
                "/path/to/data/24k/LibriSpeech_train-clean-100",
                "./svs/preprocess/make_same_speaker_dict/same_singer_LibriSpeech_train-clean-100.json",
                "LibriSpeech_train-clean-100",
            ],
            [
                "/path/to/data/24k/LibriSpeech_train-clean-360",
                "./svs/preprocess/make_same_speaker_dict/same_singer_LibriSpeech_train-clean-360.json",
                "LibriSpeech_train-clean-360",
            ],
        ],
        help="For making the dataloader that outputs source1 and source2 from SAME speaker. list of [[data_root,data_dict_path, data_name], ...]",
    )

    parser.add_argument(
        "--unison_prob",
        type=float,
        default=0.1,
        help="unison augmentation probability. If 0., no augmentation",
    )
    parser.add_argument(
        "--pitch_formant_augment_prob",
        type=float,
        default=0.2,
        help="pitch shift + formant augmentation. If 0., no augmentation",
    )

    parser.add_argument(
        "--song_length_dict_path",
        type=str,
        default="./svs/preprocess/song_length_dict.json",
        help="path of json file that contains the lengths of data",
    )

    parser.add_argument(
        "--valid_regions_dict_path",
        type=str,
        default="./svs/preprocess/valid_regions_dict_singing_singing.json",
        help="path of json file that contains the lengths of data",
    )
    parser.add_argument(
        "--seq_dur",
        type=float,
        default=4.0,
        help="Sequence duration in seconds"
        "value of <=0.0 will use full/variable length",
    )
    parser.add_argument(
        "--part_of_data",
        type=float,
        default=None,
        help="to check the effect of data amount",
    )

    args, _ = parser.parse_known_args()

    train_dataset = DuetSingingSpeechMixTraining(
        singing_data_dir=args.train_root,
        speech_data_dir=args.speech_train_root,
        song_length_dict_path=args.song_length_dict_path,
        same_song_dict_path=args.same_song_dict_path,
        same_singer_dict_path=args.same_singer_dict_path,
        same_speaker_dict_path=args.same_speaker_dict_path,
        segment=args.seq_dur,
        unison_prob=args.unison_prob,
        pitch_formant_augment_prob=args.pitch_formant_augment_prob,
        augment=True,
        part_of_data=args.part_of_data,
        sing_sing_ratio=args.sing_sing_ratio,
        sing_speech_ratio=args.sing_speech_ratio,
        same_song_ratio=args.same_song_ratio,
        same_singer_ratio=args.same_singer_ratio,
        same_speaker_ratio=args.same_speaker_ratio,
        # speech_speech_ratio=args.speech_speech_ratio
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    for epoch in range(1000):
        for mixture, sources in tqdm.tqdm(train_loader):
            print(mixture.shape)  # [32,48000]
            print(sources.shape)  # [32, 2, 48000]