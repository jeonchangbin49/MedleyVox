import numpy as np
import webrtcvad
import librosa

# Code for silence split using py-webrtcvad (https://github.com/wiseman/py-webrtcvad)
def webrtc_vad(wav, orig_sr, vad_mode=3, frame_size=0.03):
    vad = webrtcvad.Vad(vad_mode)
    if orig_sr not in [8000, 16000, 32000, 48000]:
        wav_resampled = librosa.resample(wav, orig_sr=orig_sr, target_sr=16000)
        target_sr = 16000
    else:
        wav_resampled = wav
        target_sr = orig_sr

    voice_activities = []
    for i in range(wav_resampled.shape[0] // int(16000 * frame_size)):
        voice_activities.append(
            vad.is_speech(
                wav_resampled[
                    i * int(16000 * frame_size) : (i + 1) * int(16000 * frame_size)
                ]
                .astype(np.float16)
                .tobytes(),
                target_sr,
            )
        )
    voice_activities = np.array(voice_activities) * 1

    diff = np.diff(np.pad(voice_activities, (1, 1)))

    seg_start_pos_list = np.where(diff == 1)[0]
    segment_end_pos_list = np.where(diff == -1)[0]

    return seg_start_pos_list * int(frame_size * orig_sr), segment_end_pos_list * int(
        frame_size * orig_sr
    )


# Code for silence split using the method in "Weakly Informed Source Separation, K. Shulcze-Forster, WASPAA 2019."
def magspec_vad(wav, n_fft=1024, hop_length=256):
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, center=False)
    mag, phase = librosa.magphase(stft)
    mag = mag / np.max(mag)
    mag_sum = mag.sum(0)
    mag_sum[mag_sum >= 0.1] = 1
    mag_sum[mag_sum != 1] = 0

    diff = np.diff(np.pad(mag_sum, (1, 1)))

    seg_start_pos_list = np.where(diff == 1)[0]
    segment_end_pos_list = np.where(diff == -1)[0]

    return seg_start_pos_list * hop_length, segment_end_pos_list * hop_length
