# currently not used for the inference code, but maybe for future usage?
import numpy as np


def remove_quant_noise_framewise(
    self, y, th_dB=-65, th_amp=0.0035, hopTime=0.01, fs=8000, eps=1e-14
):
    m, nch = self.calc_length_of_wav(y)
    y_truncated = y * 0
    y_idx = np.zeros((m))

    hopSize = int(np.floor(hopTime * fs))  # 3 msec hop size
    numHops = int(np.floor(m / hopSize) - 1)  # number of hops to be proccessed

    from_eff = 0
    to_eff = hopSize

    for idx_blk in np.arange(numHops - 1):
        from_ = max(0, (idx_blk) * hopSize)
        to_ = (idx_blk + 2) * hopSize
        toZ_ = (idx_blk + 1) * hopSize

        if nch == 1:
            dbBlk = 10 * np.log10(np.mean(y[from_:to_] ** 2 + eps))
            max_val = np.max(np.abs(y[from_:to_]))
        elif nch == 2:
            dbBlk = 10 * np.log10(np.mean(y[:, from_:to_] ** 2 + eps))
            max_val = np.max(np.abs(y[:, from_:to_]))

        if dbBlk > th_dB or max_val > th_amp:

            if nch == 1:
                y_truncated[from_eff:to_eff] = y[from_:toZ_]
            else:
                y_truncated[:, from_eff:to_eff] = y[:, from_:toZ_]

            y_idx[from_eff:to_eff] = np.arange(from_, toZ_)

            from_eff += hopSize
            to_eff += hopSize

    if nch == 1:
        y_truncated = y_truncated[0:m]
    else:
        y_truncated = y_truncated[:, 0:m]

    return y_truncated, y_idx


def insert_zeros(self, wav, wav_idx):
    length_of_wav = int(np.max(wav_idx))
    if len(wav.shape) == 1:
        nch = 1
        wav_out = np.zeros((length_of_wav))
    else:
        nch = 2
        wav_out = np.zeros((nch, length_of_wav))

    try:
        for idx_in, idx_out in enumerate(wav_idx):
            if idx_out > 0:
                if nch == 1:
                    wav_out[int(idx_out)] = wav[int(idx_in)]
                elif nch == 2:
                    wav_out[:, int(idx_out)] = wav[:, int(idx_in)]
    except:
        temp = 1

    return wav_out
