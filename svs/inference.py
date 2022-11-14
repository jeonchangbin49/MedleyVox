import os
import json
import argparse
import glob

import numpy as np
import soundfile as sf
import librosa
import torch
import pyloudnorm as pyln

from .models import load_model_with_args
from .functions import load_ola_func_with_args
from .utils import loudnorm, str2bool, db2linear


def main():
    parser = argparse.ArgumentParser(description="model test.py")

    parser.add_argument("--target", type=str, default="vocals")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument(
        "--suffix_name",
        type=str,
        default="",
        help="additional folder name you want to attach on the last folder name of 'exp_name'. for example, '_online'",
    )
    parser.add_argument(
        "--model_dir", type=str, default="/path/to/results/singing_sep"
    )
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument(
        "--use_overlapadd",
        type=str,
        default=None,
        choices=[None, "ola", "ola_norm", "w2v", "w2v_chunk", "sf_chunk"],
        help="use overlapadd functions, ola, ola_norm, w2v will work with ola_window_len, ola_hop_len argugments. w2v_chunk and sf_chunk is chunk-wise processing based on VAD, so you have to specify the vad_method args. If you use sf_chunk (spectral_featrues_chunk), you also need to specify spectral_features.",
    )
    parser.add_argument(
        "--vad_method",
        type=str,
        default="spec",
        choices=["spec", "webrtc"],
        help="what method do you want to use for 'voice activity detection (vad) -- split chunks -- processing. Only valid when 'w2v_chunk' or 'sf_chunk' for args.use_overlapadd.",
    )
    parser.add_argument(
        "--spectral_features",
        type=str,
        default="mfcc",
        choices=["mfcc", "spectral_centroid"],
        help="what spectral feature do you want to use in correlation calc in speaker assignment (only valid when using sf_chunk)",
    )
    parser.add_argument(
        "--w2v_ckpt_dir",
        type=str,
        default="./pretrained_models",
        help="only valid when use_overlapadd is 'w2v' or 'w2v_chunk'.",
    )
    parser.add_argument(
        "--w2v_nth_layer_output",
        nargs="+",
        type=int,
        default=[0],
        help="wav2vec nth layer output",
    )
    parser.add_argument(
        "--ola_window_len",
        type=float,
        default=None,
        help="ola window size in [sec]",
    )
    parser.add_argument(
        "--ola_hop_len",
        type=float,
        default=None,
        help="ola hop size in [sec]",
    )
    parser.add_argument(
        "--use_ema_model",
        type=str2bool,
        default=True,
        help="use ema model or online model? only vaind when args.ema it True (model trained with ema)",
    )
    parser.add_argument(
        "--save_normalized_input",
        type=str2bool,
        default=False,
        help="save normalized inputs",
    )
    parser.add_argument(
        "--stereo",
        type=str,
        default=None,
        help='if you want to inference stereo audio, choose which channel ("left" or "right") you want to use',
    )
    parser.add_argument(
        "--start_sec", type=float, default=0.0, help="start position in [sec]"
    )
    parser.add_argument(
        "--read_length", type=float, default=None, help="choose length to read."
    )
    parser.add_argument(
        "--mix_consistent_out",
        type=str2bool,
        default=True,
        help="only valid when the model is trained with mixture_consistency loss. Default is True.",
    )
    parser.add_argument(
        "--reorder_chunks",
        type=str2bool,
        default=True,
        help="ola reorder chunks",
    )
    parser.add_argument(
        "--inference_data_dir",
        type=str,
        default="./segments/24k",
        help="data where you want to separate",
    )
    parser.add_argument("--results_save_dir", type=str, default="./my_sep_results")

    args, _ = parser.parse_known_args()

    args.exp_result_dir = f"{args.model_dir}/checkpoint/{args.exp_name}"

    with open(f"{args.exp_result_dir}/{args.target}.json", "r") as f:
        args_dict = json.load(f)

    for key, value in args_dict["args"].items():
        setattr(args, key, value)

    # load model architecture
    model = load_model_with_args(args)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )
    target_model_path = f"{args.exp_result_dir}/{args.target}.pth"
    checkpoint = torch.load(target_model_path, map_location=device)
    if args.ema and args.use_ema_model:
        print("use ema model")
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        checkpoint = {
            k.replace("ema_model.module.", ""): v
            for k, v in checkpoint.items()
            if k.replace("ema_model.module.", "") in model_dict
        }
        # 2. overwrite entries in the existing state dict
        model_dict.update(checkpoint)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    elif args.ema and not args.use_ema_model:
        print("use ema online model")
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        checkpoint = {
            k.replace("online_model.module.", ""): v
            for k, v in checkpoint.items()
            if k.replace("online_model.module.", "") in model_dict
        }
        # 2. overwrite entries in the existing state dict
        model_dict.update(checkpoint)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    meter = pyln.Meter(args.sample_rate)

    if args.use_overlapadd:
        continuous_nnet = load_ola_func_with_args(args, model, device, meter)

    if args.use_overlapadd and args.mix_consistent_out:
        save_dir = f"{args.results_save_dir}/{args.exp_name}_{args.use_overlapadd}{args.suffix_name}"
    elif args.use_overlapadd and not args.mix_consistent_out:
        save_dir = f"{args.results_save_dir}/{args.exp_name}_{args.use_overlapadd}_inconsistent{args.suffix_name}"
    else:
        save_dir = f"{args.results_save_dir}/{args.exp_name}{args.suffix_name}"

    os.makedirs(f"{save_dir}", exist_ok=True)

    data_list = (
        glob.glob(f"{args.inference_data_dir}/*.wav")
        + glob.glob(f"{args.inference_data_dir}/*.mp3")
        + glob.glob(f"{args.inference_data_dir}/*.flac")
    )
    print(data_list)

    for data_path in data_list:
        song_name = (
            os.path.basename(data_path)
            .replace(".wav", "")
            .replace(".mp3", "")
            .replace(".flac", "")
        )
        print(f"now separating {song_name}")

        if args.stereo == "left":
            mixture, sr = librosa.load(
                data_path,
                sr=args.sample_rate,
                mono=False,
                offset=args.start_sec,
                duration=args.read_length,
                dtype=np.float32,
            )
            mixture = mixture[0, :]
        elif args.stereo == "right":
            mixture, sr = librosa.load(
                data_path,
                sr=args.sample_rate,
                mono=False,
                offset=args.start_sec,
                duration=args.read_length,
                dtype=np.float32,
            )
            mixture = mixture[1, :]
        else:
            mixture, sr = librosa.load(
                data_path,
                sr=args.sample_rate,
                mono=True,
                offset=args.start_sec,
                duration=args.read_length,
                dtype=np.float32,
            )

        mixture, adjusted_gain = loudnorm(mixture, -24.0, meter)

        if args.save_normalized_input:
            os.makedirs(f"{args.inference_data_dir}/24k_normalized", exist_ok=True)
            sf.write(
                f"{args.inference_data_dir}/24k_normalized/{song_name}.wav", mixture, sr
            )

        mixture = np.expand_dims(mixture, axis=0)
        mixture = mixture.reshape(1, mixture.shape[0], mixture.shape[1])
        mixture = torch.as_tensor(mixture, dtype=torch.float32).to(device)

        if args.use_overlapadd:
            out_wavs = continuous_nnet.forward(mixture)
        else:
            out_wavs = model.separate(mixture)

        if args.use_gpu:
            out_wav_1 = out_wavs[0, 0, :].cpu().detach().numpy()
            out_wav_2 = out_wavs[0, 1, :].cpu().detach().numpy()
        else:
            out_wav_1 = out_wavs[0, 0, :]
            out_wav_2 = out_wavs[0, 1, :]

        out_wav_1 = out_wav_1 * db2linear(-adjusted_gain)
        out_wav_2 = out_wav_2 * db2linear(-adjusted_gain)

        if args.start_sec != 0.0:  # for sync between the output and the input audio.
            out_wav_1 = np.pad(
                out_wav_1, (int(args.start_sec * args.sample_rate) - 1, 0)
            )
            out_wav_2 = np.pad(
                out_wav_2, (int(args.start_sec * args.sample_rate) - 1, 0)
            )

        if args.stereo:
            save_wav_path_1 = f"{save_dir}/{song_name}_output_{args.stereo}_1.wav"
            save_wav_path_2 = f"{save_dir}/{song_name}_output_{args.stereo}_2.wav"
        else:
            save_wav_path_1 = f"{save_dir}/{song_name}_output_1.wav"
            save_wav_path_2 = f"{save_dir}/{song_name}_output_2.wav"

        sf.write(
            save_wav_path_1,
            out_wav_1,
            args.sample_rate,
        )
        sf.write(
            save_wav_path_2,
            out_wav_2,
            args.sample_rate,
        )


if __name__ == "__main__":
    main()
