import random
import json

import numpy as np
import torch
from torch.utils.data import Dataset
import pyloudnorm as pyln

from ..utils import (
    load_wav_arbitrary_position_mono,
    load_wav_specific_position_mono,
    db2linear,
    loudness_match_and_norm,
    loudness_normal_match_and_norm_output_louder_first,
    loudnorm,
    change_pitch_and_formant,
)
from .singing_libri_dataset import DuetSingingSpeechMixTraining


# Singing dataset + LibriSpeech dataset

# load singing_only list
# load LibriSpeech list separately


class MultiSingingSpeechMixTraining(DuetSingingSpeechMixTraining):
    """Dataset class for multiple singing voice separation tasks.

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

    dataset_name = "multi_singing_with_speech"

    def __init__(
        self,
        singing_data_dir,
        speech_data_dir,
        song_length_dict_path,
        same_song_dict_path,
        same_singer_dict_path,
        same_speaker_dict_path,
        min_n_src=3,
        max_n_src=10,
        sample_rate=24000,
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
        super().__init__(
            singing_data_dir,
            speech_data_dir,
            song_length_dict_path,
            same_song_dict_path,
            same_singer_dict_path,
            same_speaker_dict_path,
            sample_rate,
            n_src,
            segment,
            unison_prob,
            pitch_formant_augment_prob,
            augment,
            part_of_data,
            reduced_training_data_ratio,
            sing_sing_ratio,
            sing_speech_ratio,
            same_song_ratio,
            same_singer_ratio,
            same_speaker_ratio,
        )
        self.min_n_src = min_n_src  # minimum number of sources in mixture
        self.max_n_src = max_n_src  # maximum number of sources in mixture
        # if min_n_src and max_n_src are identical, fixed number of sources will be sampled for making a mixture

    def __len__(self):
        return int(self.len_singing_train_list * self.reduced_training_data_ratio)

    def load_additional_audio(self, source_2, rand_prob, paths_info_dict):
        # load additional data
        additional_n_src = (
            random.randint(self.min_n_src, self.max_n_src) - 2
        )  # minus 2 because we already sampled 2 sources
        # Case 1. different singing + singing + ...
        if rand_prob <= self.sing_sing_ratio_cum:
            for i in range(additional_n_src):
                data_path_n = random.choice(self.singing_train_list)
                source_2 = source_2 + load_wav_arbitrary_position_mono(
                    data_path_n, self.sample_rate, self.segment
                ) * random.uniform(0.25, 1.25)
        # Case 2. singing + speech + singing + speech + ...
        elif self.sing_sing_ratio_cum < rand_prob <= self.sing_speech_ratio_cum:
            for i in range(additional_n_src):
                if i % 2 == 0:
                    data_path_n = random.choice(self.speech_wav_paths)
                else:
                    data_path_n = random.choice(self.singing_train_list)
                source_2 = source_2 + load_wav_arbitrary_position_mono(
                    data_path_n, self.sample_rate, self.segment
                ) * random.uniform(0.25, 1.25)
        # Case 3. same song (but with different singer)
        elif self.sing_speech_ratio_cum < rand_prob <= self.same_song_ratio_cum:
            for i in range(additional_n_src):
                if (
                    len(paths_info_dict["same_list"]) == 0
                ):  # no more song left to sample
                    data_path_n = random.choice(self.singing_train_list)
                    source_2 = source_2 + load_wav_arbitrary_position_mono(
                        data_path_n, self.sample_rate, self.segment
                    ) * random.uniform(0.25, 1.25)
                else:  # some songs are left to sample
                    filename_n = random.choice(paths_info_dict["same_list"])
                    data_path_n = f"{paths_info_dict['data_root']}/{filename_n[0]}.wav"
                    paths_info_dict["same_list"].remove(filename_n)
                    # try:
                    source_2 = source_2 + load_wav_specific_position_mono(
                        data_path_n,
                        self.sample_rate,
                        self.segment,
                        paths_info_dict["rand_start"],
                    ) * random.uniform(0.25, 1.25)

        # Case 4. same singer (but with different song)
        elif self.same_song_ratio_cum < rand_prob <= self.same_singer_ratio_cum:
            for i in range(additional_n_src):
                if len(paths_info_dict["same_list"]) == 0:
                    data_path_n = random.choice(self.singing_train_list)
                    source_2 = source_2 + load_wav_arbitrary_position_mono(
                        data_path_n, self.sample_rate, self.segment
                    ) * random.uniform(0.25, 1.25)
                else:
                    filename_n = random.choice(paths_info_dict["same_list"])
                    data_path_n = f"{paths_info_dict['data_root']}/{filename_n}.wav"
                    paths_info_dict["same_list"].remove(filename_n)
                    source_2 = source_2 + load_wav_arbitrary_position_mono(
                        data_path_n, self.sample_rate, self.segment
                    ) * random.uniform(0.25, 1.25)

        # Case 5. same speaker (but with different speech)
        elif self.same_singer_ratio_cum < rand_prob <= self.same_speaker_ratio_cum:
            for i in range(additional_n_src):
                if len(paths_info_dict["same_list"]) == 0:
                    data_path_n = random.choice(self.speech_wav_paths)
                    source_2 = source_2 + load_wav_arbitrary_position_mono(
                        data_path_n, self.sample_rate, self.segment
                    ) * random.uniform(0.25, 1.25)
                else:
                    filename_n = random.choice(paths_info_dict["same_list"])
                    data_path_n = f"{paths_info_dict['data_root']}/{filename_n}.wav"
                    paths_info_dict["same_list"].remove(filename_n)
                    source_2 = source_2 + load_wav_arbitrary_position_mono(
                        data_path_n, self.sample_rate, self.segment
                    ) * random.uniform(0.25, 1.25)

        # Case 6. different speech + speech
        elif self.same_speaker_ratio_cum < rand_prob <= 1:
            data_path_n = random.choice(self.speech_wav_paths)
            source_2 = source_2 + load_wav_arbitrary_position_mono(
                data_path_n, self.sample_rate, self.segment
            ) * random.uniform(0.25, 1.25)

        return source_2

    def __getitem__(self, idx):
        rand_prob = random.random()

        paths_info_dict = self.return_paths(idx, rand_prob)

        # Load audio
        sources_list = []

        source_1, paths_info_dict = self.load_first_audio(rand_prob, paths_info_dict)

        augment_prob = random.random()
        source_1, source_2 = self.load_second_audio(
            source_1, rand_prob, augment_prob, paths_info_dict
        )

        source_2 = source_2 * random.uniform(
            0.25, 1.25
        )  # random scaling of source_2 (dynamic mixing)

        source_2 = self.load_additional_audio(source_2, rand_prob, paths_info_dict)

        if self.augment:
            source_1, source_2 = loudness_normal_match_and_norm_output_louder_first(
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


class MultiSingingSpeechMixValidation(Dataset):
    """Dataset class for multip singing voice separation. For validation dataset
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

    dataset_name = "multi_singing_with_speech_valid"

    def __init__(
        self,
        data_dir,
        sample_rate=24000,
        n_src=2,
        segment=6,
        augment=True,
    ):
        self.source_1_paths = []
        self.source_2_data_root = []
        self.metadata_list = []

        for data_dir_set in data_dir:

            with open(data_dir_set[2], "r") as json_file:
                self.valid_regions_dict = json.load(json_file)
            for key, value in self.valid_regions_dict.items():
                self.source_1_paths.append(f"{data_dir_set[0]}/{key}")
                if data_dir_set[3] == "singing_speech":
                    self.source_2_data_root.append([data_dir_set[0], data_dir_set[1]])
                else:
                    self.source_2_data_root.append(data_dir_set[1])
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
        metadata = self.metadata_list[idx]

        sources_list = []

        source_1 = load_wav_specific_position_mono(
            data_path_1, self.sample_rate, self.segment, 0.0
        )  # data_1 starts from 0. sec
        source_2 = 0.0
        for src_idx, src_basename in enumerate(
            metadata["corresponding_data"]
        ):  # metadata["corresponding_data"] contains the list of file names
            if metadata["type"] == "singing+speech":
                if src_idx % 2 == 0:
                    source_2_temp_file_path = (
                        f"{self.source_2_data_root[idx][1]}/{src_basename}"
                    )
                else:
                    source_2_temp_file_path = (
                        f"{self.source_2_data_root[idx][0]}/{src_basename}"
                    )
            else:
                source_2_temp_file_path = (
                    f"{self.source_2_data_root[idx]}/{src_basename}"
                )
            source_2_temp = load_wav_specific_position_mono(
                source_2_temp_file_path,
                self.sample_rate,
                self.segment,
                metadata["position(sec)"][src_idx],
            )

            if metadata["unison_aug"]:
                source_2_temp = change_pitch_and_formant(
                    source_2_temp,
                    self.sample_rate,
                    metadata["unison_params"][src_idx][0],
                    metadata["unison_params"][src_idx][1],
                    1,
                    metadata["unison_params"][src_idx][3],
                )
            source_2_temp = (
                source_2_temp * metadata["gain_adjustment"][src_idx]
            )  # gain scaling of source_2_temp
            source_2 = source_2_temp

        if self.augment:
            source_1, source_2 = loudness_match_and_norm(
                source_1, source_2, self.meter
            )  # match the loudness of the sources

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
        # if not self.return_id:

        # return mixture, sources, metadata[type]
        return mixture, sources
