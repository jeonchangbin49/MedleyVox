import os
import glob
import json

import numpy as np
import torch
from torch.utils.data import Dataset
import librosa


class MedleyVox(Dataset):
    """Dataset class for MedleyVox source separation tasks.

    Args:
        task (str): One of ``'unison'``, ``'duet'``, ``'main_vs_rest'`` or
            ``'total'`` :
            * ``'unison'`` for unison vocal separation.
            * ``'duet'`` for duet vocal separation.
            * ``'main_vs_rest'`` for main vs. rest vocal separation (main vs rest).
            * ``'n_singing'`` for N-singing separation. We will use all of the duet, unison, and main vs. rest data.

        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture. Actually, this is fixed to 2 for our tasks. Need to be specified for N-singing training (future work).
        segment (int, optional) : The desired sources and mixtures length in s.
    """

    dataset_name = "MedleyVox"

    def __init__(
        self,
        root_dir,
        metadata_dir=None,
        task="duet",
        sample_rate=24000,
        n_src=2,
        segment=None,
        return_id=True,
    ):
        self.root_dir = root_dir  # /path/to/data/test_medleyDB
        self.metadata_dir = metadata_dir  # ./testset/testset_config
        self.task = task.lower()
        self.return_id = return_id
        # Get the csv corresponding to the task
        if self.task == "unison":
            self.total_segments_list = glob.glob(f"{self.root_dir}/unison/*/*")
        elif self.task == "duet":
            self.total_segments_list = glob.glob(f"{self.root_dir}/duet/*/*")
        elif self.task == "main_vs_rest":
            self.total_segments_list = glob.glob(f"{self.root_dir}/rest/*/*")
        elif self.task == "n_singing":
            self.total_segments_list = glob.glob(f"{self.root_dir}/unison/*/*") + glob.glob(f"{self.root_dir}/duet/*/*") + glob.glob(f"{self.root_dir}/rest/*/*")
        self.segment = segment
        self.sample_rate = sample_rate
        self.n_src = n_src

    def __len__(self):
        return len(self.total_segments_list)

    def __getitem__(self, idx):
        song_name = self.total_segments_list[idx].split("/")[-2]
        segment_name = self.total_segments_list[idx].split("/")[-1]
        mixture_path = (
            f"{self.total_segments_list[idx]}/mix/{song_name} - {segment_name}.wav"
        )
        self.mixture_path = mixture_path
        sources_path_list = glob.glob(f"{self.total_segments_list[idx]}/gt/*.wav")

        if self.task == "main_vs_rest" or self.task == "n_singing":
            if os.path.exists(
                f"{self.metadata_dir}/V1_rest_vocals_only_config/{song_name}.json"
            ):
                metadata_json_path = (
                    f"{self.metadata_dir}/V1_rest_vocals_only_config/{song_name}.json"
                )
            elif os.path.exists(
                f"{self.metadata_dir}/V2_vocals_only_config/{song_name}.json"
            ):
                metadata_json_path = (
                    f"{self.metadata_dir}/V2_vocals_only_config/{song_name}.json"
                )
            else:
                print("main vs. rest metadata not found.")
                raise AttributeError
            with open(metadata_json_path, "r") as json_file:
                metadata_json = json.load(json_file)

        # Read sources
        sources_list = []
        ids = []
        if self.task == "main_vs_rest" or self.task == "n_singing":
            gt_main_name = metadata_json[segment_name]["main_vocal"]
            gt_source, sr = librosa.load(
                f"{self.total_segments_list[idx]}/gt/{gt_main_name} - {segment_name}.wav",
                sr=self.sample_rate,
            )
            gt_rest_list = metadata_json[segment_name]["other_vocals"]
            ids.append(f"{gt_main_name} - {segment_name}")

            rest_sources_list = []
            for other_vocal_name in gt_rest_list:
                s, sr = librosa.load(
                    f"{self.total_segments_list[idx]}/gt/{other_vocal_name} - {segment_name}.wav",
                    sr=self.sample_rate,
                )
                rest_sources_list.append(s)
                ids.append(f"{other_vocal_name} - {segment_name}")
            rest_sources_list = np.stack(rest_sources_list, axis=0)
            gt_rest = rest_sources_list.sum(0)

            sources_list.append(gt_source)
            sources_list.append(gt_rest)
        else: # self.task == 'unison' or self.task == 'duet'
            for i, source_path in enumerate(sources_path_list):
                s, sr = librosa.load(source_path, sr=self.sample_rate)
                sources_list.append(s)
                ids.append(os.path.basename(source_path).replace(".wav", ""))
        # Read the mixture
        mixture, sr = librosa.load(mixture_path, sr=self.sample_rate)
        # Convert to torch tensor
        mixture = torch.as_tensor(mixture, dtype=torch.float32)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.as_tensor(sources, dtype=torch.float32)
        if not self.return_id:
            return mixture, sources
        # 5400-34479-0005_4973-24515-0007.wav
        # id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")

        return mixture, sources, ids
