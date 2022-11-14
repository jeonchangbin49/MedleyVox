import random

import numpy as np
import librosa
import torchaudio


def load_wav_arbitrary_position_mono(filename, sample_rate, seq_duration):
    # mono
    # seq_duration[second]
    length = torchaudio.info(filename).num_frames

    read_length = librosa.time_to_samples(seq_duration, sr=sample_rate)
    if length > read_length:
        random_start = random.randint(0, int(length - read_length - 1)) / sample_rate
        X, sr = librosa.load(
            filename, sr=None, offset=random_start, duration=seq_duration
        )
    else:
        random_start = 0
        total_pad_length = read_length - length
        X, sr = librosa.load(filename, sr=None, offset=0, duration=seq_duration)
        pad_left = random.randint(0, total_pad_length)
        X = np.pad(X, (pad_left, total_pad_length - pad_left))

    return X


def load_wav_specific_position_mono(
    filename, sample_rate, seq_duration, start_position
):
    # mono
    # seq_duration[second]
    # start_position[second]
    length = torchaudio.info(filename).num_frames
    read_length = librosa.time_to_samples(seq_duration, sr=sample_rate)

    start_pos_sec = max(
        start_position, 0
    )  # if start_position is minus, then start from 0.
    start_pos_sample = librosa.time_to_samples(start_pos_sec, sr=sample_rate)

    if (
        length <= start_pos_sample
    ):  # if start position exceeds audio length, then start from 0.
        start_pos_sec = 0
        start_pos_sample = 0
    X, sr = librosa.load(filename, sr=None, offset=start_pos_sec, duration=seq_duration)

    if length < start_pos_sample + read_length:
        X = np.pad(X, (0, (start_pos_sample + read_length) - length))

    return X
