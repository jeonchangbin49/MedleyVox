import librosa
import numpy as np


def ideal_ratio_mask(args, mix, sources):
    """
    Ideal Ratio Mask (IRM)
    """
    mix_stft = librosa.stft(mix, n_fft=args.nfft, hop_length=args.nhop)
    mix_mag, mix_phase = librosa.magphase(mix_stft, power=1)

    source_1_stft = librosa.stft(sources[0], n_fft=args.nfft, hop_length=args.nhop)
    source_1_mag, source_1_phase = librosa.magphase(source_1_stft, power=1)

    source_2_stft = librosa.stft(sources[1], n_fft=args.nfft, hop_length=args.nhop)
    source_2_mag, source_2_phase = librosa.magphase(source_2_stft, power=1)

    irm_1 = source_1_mag / (source_1_mag + source_2_mag + 1e-8)
    irm_2 = source_2_mag / (source_1_mag + source_2_mag + 1e-8)

    irm_1_output = mix_mag * irm_1 * mix_phase
    irm_2_output = mix_mag * irm_2 * mix_phase

    irm_1_output = librosa.istft(
        irm_1_output, win_length=args.nfft, hop_length=args.nhop, n_fft=args.nfft
    )
    irm_2_output = librosa.istft(
        irm_2_output, win_length=args.nfft, hop_length=args.nhop, n_fft=args.nfft
    )

    return np.stack([irm_1_output, irm_2_output], axis=0)


def ideal_binary_mask(args, mix, sources):
    """
    Ideal Binary Mask (IBM)
    """
    mix_stft = librosa.stft(mix, n_fft=args.nfft, hop_length=args.nhop)
    mix_mag, mix_phase = librosa.magphase(mix_stft, power=1)

    source_1_stft = librosa.stft(sources[0], n_fft=args.nfft, hop_length=args.nhop)
    source_1_mag, source_1_phase = librosa.magphase(source_1_stft, power=1)

    source_2_stft = librosa.stft(sources[1], n_fft=args.nfft, hop_length=args.nhop)
    source_2_mag, source_2_phase = librosa.magphase(source_2_stft, power=1)

    ibm_1 = np.zeros_like(source_1_mag)
    ibm_2 = np.zeros_like(source_2_mag)

    ibm_1[source_1_mag > source_2_mag] = 1
    ibm_2[source_2_mag > source_1_mag] = 1

    ibm_1_output = mix_mag * ibm_1 * mix_phase
    ibm_2_output = mix_mag * ibm_2 * mix_phase

    ibm_1_output = librosa.istft(
        ibm_1_output, win_length=args.nfft, hop_length=args.nhop, n_fft=args.nfft
    )
    ibm_2_output = librosa.istft(
        ibm_2_output, win_length=args.nfft, hop_length=args.nhop, n_fft=args.nfft
    )

    return np.stack([ibm_1_output, ibm_2_output], axis=0)


# cirm is based on https://gist.github.com/jonashaag/677e1ddab99f3daba367de9ec022e942
def cirm_out(y, s, K=1, C=0.1, flat=False):
    y = librosa.core.stft(y.astype("float64"), 2048, 512).astype("complex128")
    s = librosa.core.stft(s.astype("float64"), 2048, 512).astype("complex128")
    mr = (np.real(y) * np.real(s) + np.imag(y) * np.imag(s)) / (
        np.real(y) ** 2 + np.imag(y) ** 2
    )
    mi = (np.real(y) * np.imag(s) - np.imag(y) * np.real(s)) / (
        np.real(y) ** 2 + np.imag(y) ** 2
    )
    m = mr + 1j * mi
    if flat:
        return librosa.istft(y * m, win_length=2048, hop_length=512, n_fft=2048)
    else:
        return librosa.istft(
            y * inverse_mask(K * ((1 - np.exp(-C * m)) / (1 + np.exp(-C * m))), m),
            win_length=2048,
            hop_length=512,
            n_fft=2048,
        )


def inverse_mask(x, m, K=1, C=0.1, flat=False):
    if flat:
        return x * m
    else:
        return -1 / C * np.log((K - x) / (K + x + 1e-8) + 1e-8)


def complex_ideal_ratio_mask(args, mix, sources):
    """
    Complex Ideal Ratio Mask (cIRM)
    """
    source_1_out = cirm_out(mix, sources[0])
    source_2_out = cirm_out(mix, sources[1])

    return np.stack([source_1_out, source_2_out], axis=0)


def return_oracle_with_args(args, mix, sources):
    if args.oracle_method == "irm":
        out_sources = ideal_ratio_mask(args, mix, sources)
    elif args.oracle_method == "ibm":
        out_sources = ideal_binary_mask(args, mix, sources)
    elif args.oracle_method == "cirm":
        out_sources = complex_ideal_ratio_mask(args, mix, sources)

    out_sources = np.pad(
        out_sources, ((0, 0), (0, sources.shape[1] - out_sources.shape[1]))
    )

    return out_sources
