import os
import random
import json
from tqdm import tqdm
from pprint import pprint

import soundfile as sf
import argparse
import pandas as pd
from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix

from .data import MedleyVox
from .functions import return_oracle_with_args

COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(args):
    compute_metrics = COMPUTE_METRICS

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

    eval_save_dir = (
        f"{args.exp_dir}/{args.out_dir}/{args.oracle_method}{args.suffix_name}"
    )
    ex_save_dir = f"{eval_save_dir}/examples/"
    if args.n_save_ex == -1:
        args.n_save_ex = len(test_set)
    save_idx = random.sample(range(len(test_set)), args.n_save_ex)
    series_list = []

    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources, ids = test_set[idx]
        mix, sources = mix.numpy(), sources.numpy()
        est_sources = return_oracle_with_args(args, mix, sources)

        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        utt_metrics = get_metrics(
            mix,
            sources,
            est_sources,
            sample_rate=args.sample_rate,
            metrics_list=COMPUTE_METRICS,
        )
        print("utt_metrics", utt_metrics)
        utt_metrics["mix_path"] = test_set.mixture_path

        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = f"{ex_save_dir}/ex_{idx}/"
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix, args.sample_rate)
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources):
                sf.write(f"{local_save_dir}/s{src_idx}.wav", src, args.sample_rate)
            for src_idx, est_src in enumerate(est_sources):
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
    parser.add_argument("--oracle_method", type=str, default="irm")
    parser.add_argument(
        "--suffix_name",
        type=str,
        default="",
        help="additional folder name you want to attach on the last folder name of 'exp_name'. for example, '_online'",
    )

    # Original parameters of test code
    parser.add_argument(
        "--test_dir",
        type=str,
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
        default="sep_clean",
        help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="eval_results",
        help="Directory in exp_dir where the eval results" " will be stored",
    )
    parser.add_argument(
        "--exp_dir",
        default="/path/to/results/singing_sep",
        help="Experiment root",
    )
    parser.add_argument(
        "--n_save_ex",
        type=int,
        default=-1,
        help="Number of audio examples to save, -1 means all",
    )
    parser.add_argument("--nfft", type=int, default=2048)
    parser.add_argument("--nhop", type=int, default=512)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--n_src", type=int, default=24000)

    args, _ = parser.parse_known_args()

    main(args)
