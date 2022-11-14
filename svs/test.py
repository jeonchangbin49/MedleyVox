import os
import random
import json
from tqdm import tqdm
from pprint import pprint

import soundfile as sf
import librosa
import numpy as np
import torch
import argparse
import pandas as pd
import pyloudnorm as pyln
from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.utils import tensors_to_device

from .data import MedleyVox
from .models import load_model_with_args
from .functions import load_ola_func_with_args
from .utils import str2bool, loudnorm, db2linear

COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(args):
    compute_metrics = COMPUTE_METRICS

    # Handle device placement
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )
    model_path = f"{args.exp_result_dir}/{args.target}.pth"
    model = load_model_with_args(args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
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

    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    if args.test_target == "speech":
        test_set = LibriMix(
            csv_dir=args.test_dir,
            task=args.speech_task,
            sample_rate=args.sample_rate,
            n_src=args.n_src,
            segment=None,
            return_id=True,
        )  # Uses all segment length
    elif args.test_target == "singing":
        test_set = MedleyVox(
            root_dir=args.test_dir,
            metadata_dir=args.metadata_dir,
            task=args.singing_task,
            sample_rate=args.sample_rate,
            n_src=args.n_src,
            segment=None,
            return_id=True,
        )

    meter = pyln.Meter(args.sample_rate)

    # Define overlap add functions
    if args.use_overlapadd:
        continuous_nnet = load_ola_func_with_args(args, model, device, meter)
        eval_save_dir = f"{args.exp_dir}/{args.out_dir}/{args.exp_name}_{args.use_overlapadd}{args.suffix_name}"
    else:
        eval_save_dir = (
            f"{args.exp_dir}/{args.out_dir}/{args.exp_name}{args.suffix_name}"
        )

    ex_save_dir = f"{eval_save_dir}/examples/"
    if args.n_save_ex == -1:
        args.n_save_ex = len(test_set)

    # Randomly choose the indexes of sentences to save.
    save_idx = random.sample(range(len(test_set)), args.n_save_ex)
    series_list = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_set))):
            # Forward the network on the mixture.
            mix, sources, ids = test_set[idx]

            # Apply loudness normalization
            mix, adjusted_gain = loudnorm(mix.numpy(), -24.0, meter, eps=0.0)
            mix = torch.as_tensor(mix, dtype=torch.float32)
            sources = sources.numpy() * db2linear(adjusted_gain, eps=0.0)
            sources = torch.as_tensor(sources, dtype=torch.float32)

            mix, sources = tensors_to_device([mix, sources], device=device)
            if args.use_overlapadd:
                est_sources = continuous_nnet(mix.unsqueeze(0).unsqueeze(0))
            else:
                est_sources = model(mix.unsqueeze(0))
            loss, reordered_sources = loss_func(
                est_sources, sources[None], return_est=True
            )
            mix_np = mix.cpu().data.numpy() * db2linear(-adjusted_gain, eps=0.0)
            sources_np = sources.cpu().data.numpy() * db2linear(-adjusted_gain, eps=0.0)
            est_sources_np = reordered_sources.squeeze(
                0
            ).cpu().data.numpy() * db2linear(
                -adjusted_gain, eps=0.0
            )  # [n_src, wav_length]

            if args.save_and_load_eval:
                for source_index in range(est_sources_np.shape[0]):
                    local_save_dir = f"{ex_save_dir}{ids[0]} - {ids[1]}"
                    os.makedirs(local_save_dir, exist_ok=True)
                    if args.save_smaller_output:
                        est_sources_np[source_index], adj_gain = loudnorm(
                            est_sources_np[source_index], -27.0, meter, eps=0.0
                        )
                    sf.write(
                        f"{local_save_dir}/{ids[source_index]}_estimate.wav",
                        est_sources_np[source_index],
                        args.sample_rate,
                    )
                    est_sources_np[source_index, :] = librosa.load(
                        f"{local_save_dir}/{ids[source_index]}_estimate.wav",
                        sr=args.sample_rate,
                    )[0]
            # For each utterance, we get a dictionary with the mixture path,
            # the input and output metrics
            utt_metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=args.sample_rate,
                metrics_list=COMPUTE_METRICS,
            )
            utt_metrics["mix_path"] = test_set.mixture_path

            series_list.append(pd.Series(utt_metrics))

            # Save some examples in a folder. Wav files and metrics as text.
            if idx in save_idx:
                local_save_dir = f"{ex_save_dir}/ex_{idx}/"
                os.makedirs(local_save_dir, exist_ok=True)
                sf.write(local_save_dir + "mixture.wav", mix_np, args.sample_rate)
                # Loop over the sources and estimates
                for src_idx, src in enumerate(sources_np):
                    sf.write(f"{local_save_dir}/s{src_idx}.wav", src, args.sample_rate)
                for src_idx, est_src in enumerate(est_sources_np):
                    sf.write(
                        f"{local_save_dir}/s{src_idx}_estimate.wav",
                        est_src,
                        args.sample_rate,
                    )
                # Write local metrics to the example folder.
                with open(local_save_dir + "metrics.json", "w") as f:
                    json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(f"{eval_save_dir}/all_metrics.csv")

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print("Overall metrics :")
    print(f"{args.exp_name}{args.suffix_name}")
    pprint(final_results)

    with open(f"{eval_save_dir}/final_metrics.json", "w") as f:
        json.dump(final_results, f, indent=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model test.py")
    # Added arguments
    parser.add_argument("--target", type=str, default="vocals")
    parser.add_argument(
        "--test_target",
        type=str,
        default="singing",
        choices=["speech", "singing"],
        help="choose",
    )
    parser.add_argument(
        "--singing_task",
        type=str,
        default="duet",
        help="only valid when test_target=='singing'. 'unison' or 'duet' or 'main_vs_rest'",
    )
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
        help="only valid when use_overlapadd is 'w2v or 'w2v_chunk'.",
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
        "--reorder_chunks",
        type=str2bool,
        default=True,
        help="ola reorder chunks",
    )
    parser.add_argument(
        "--use_ema_model",
        type=str2bool,
        default=True,
        help="use ema model or online model? only vaind when args.ema it True (model trained with ema)",
    )

    # Original parameters of test code
    parser.add_argument(
        "--test_dir",
        type=str,
        # required=True,
        # default="/path/to/dataLibri2Mix/wav16k/max/metadata",
        # default="/path/to/dataLibri2Mix/wav24k/max/metadata",
        default="/path/to/data/test_medleyDB",
        help="Test directory including the csv files",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="./testset/testset_config",
        help="Metadata for testset, only for 'main vs. rest' separation",
    )
    parser.add_argument(
        "--speech_task",
        type=str,
        # required=True,
        default="sep_clean",
        help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        # required=True,
        default="eval_results",
        help="Directory in exp_dir where the eval results" " will be stored",
    )
    parser.add_argument(
        "--exp_dir",
        default="/path/to/results/singing_sep",
        help="Experiment root. Evaluation results will saved in '(args.exp_dir)/(args.out_dir)/(args.exp_name'",
    )
    parser.add_argument(
        "--n_save_ex",
        type=int,
        default=10,
        help="Number of audio examples to save, -1 means all",
    )

    parser.add_argument(
        "--save_and_load_eval",
        type=str2bool,
        default=False,
        help="To check the output scale exploding, save and load outputs for eval.",
    )
    parser.add_argument(
        "--save_smaller_output",
        type=str2bool,
        default=False,
        help="To check the output scale exploding, save and load outputs for eval.",
    )

    # Original arguments

    args, _ = parser.parse_known_args()

    args.exp_result_dir = f"{args.model_dir}/checkpoint/{args.exp_name}"

    with open(f"{args.exp_result_dir}/{args.target}.json", "r") as f:
        args_dict = json.load(f)

    for key, value in args_dict["args"].items():
        setattr(args, key, value)

    main(args)
