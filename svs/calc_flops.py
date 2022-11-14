import time
import json
import argparse

import torch
import torch.nn as nn
from deepspeed.profiling.flops_profiler import get_model_profile


from .models.super_resolution_net import SFSRNet, SFSRNet_ConvNext
from .utils import str2bool

parser = argparse.ArgumentParser(description="Singing separation Trainer")

parser.add_argument(
    "--mixed_precision",
    type=str2bool,
    default=False,
    help="use mixed precision training?",
)
parser.add_argument(
    "--gradient_clip",
    type=float,
    default=None,
    help="grad_clip max_norm parameter. None if you do not want to use grad_clip.",
)

parser.add_argument(
    "--architecture",
    type=str,
    default="conv_tasnet_stft",
    help="demucs, tasnet, trunet, umx",
)
parser.add_argument("--mask_act", type=str, default="relu", help="relu, linear")
parser.add_argument(
    "--ff_activation", type=str, default="relu", help="relu, linear, gelu, etc."
)
parser.add_argument("--encoder_activation", type=str, default=None, help="relu, linear")
parser.add_argument(
    "--no_mask",
    type=str2bool,
    default=False,
    help="use masking method? Default:False",
)
parser.add_argument(
    "--no_mask_residual",
    type=str2bool,
    default=False,
    help="use masking method? Default:False",
)
parser.add_argument(
    "--mixture_consistency",
    type=str,
    default=None,
    help="use mixture consistency training? or SFSRNet training? ['residual', 'mixture_consistency', 'sfsrnet']",
)

parser.add_argument(
    "--srnet",
    type=str,
    default="orig",
    help="use orig srnet or ConvNext style srnet? ['orig', 'convnext']",
)
parser.add_argument(
    "--db_normalize",
    type=str2bool,
    default=False,
    help="when using sfsrnet, use db_normalize of SFSRNet input?",
)
parser.add_argument(
    "--sr_input_res",
    type=str2bool,
    default=False,
    help="when using sfsrnet, use output residual connection? recommended when using original style SRNet",
)

# Network parameters
parser.add_argument(
    "--n_blocks",
    type=int,
    default=8,
    help="Number of convolutional blocks in each repeat. Defaults to 8.",
)
parser.add_argument(
    "--n_repeats", type=int, default=3, help="Number of repeats. Defaults to 3."
)
parser.add_argument(
    "--bn_chan",
    type=int,
    default=128,
    help="Number of channels after the bottleneck.",
)
parser.add_argument(
    "--skip_chan",
    type=int,
    default=128,
    help="Number of channels of skip connection outputs.",
)
parser.add_argument(
    "--hid_chan",
    type=int,
    default=512,
    help="Number of channels in the convolutional blocks.",
)

parser.add_argument(
    "--min_n_src",
    type=int,
    default=3,
    help="minimum number of sources in mixture during OR-PIT trianing",
)
parser.add_argument(
    "--max_n_src",
    type=int,
    default=10,
    help="maximum number of sources in mixture during OR-PIT trianing",
)

parser.add_argument("--ema", type=str2bool, default=False, help="use model ema?")
parser.add_argument(
    "--start_from_best",
    type=str2bool,
    default=False,
    help="when use --continual_train, do you want to start from previous best-performed weight?",
)

# Trainig Parameters
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--eps", type=float, default=1e-8, help="optimizer eps parameter")
parser.add_argument(
    "--above_freq",
    type=float,
    default=1000.0,
    help="if you want to calc spectral loss only above --above_freq. Only valid when using multi_spectral_l1_above_freq in train",
)
parser.add_argument(
    "--multi_spec_loss_log_scale",
    type=str2bool,
    default=False,
    help="use log to increase multispectral loss scale",
)

parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay")
parser.add_argument(
    "--seed", type=int, default=777, metavar="S", help="random seed (default: 42)"
)

parser.add_argument(
    "--sample_rate",
    type=int,
    default=16000,
    help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
)

parser.add_argument(
    "--seq_dur",
    type=float,
    default=3.0,
    help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
)
parser.add_argument("--n_src", type=int, default=2, help="number of estimating sources")
parser.add_argument(
    "--nfft", type=int, default=1024, help="STFT fft size and window size"
)
parser.add_argument("--nhop", type=int, default=256, help="STFT hop size")
parser.add_argument(
    "--n_filter", type=int, default=256, help="learnable basis filter size"
)
parser.add_argument(
    "--n_kernel", type=int, default=256, help="learnable basis kernel size"
)


args, _ = parser.parse_known_args()

with torch.cuda.device(0):
    model = SFSRNet()
    # model = SFSRNet_ConvNext()

    batch_size = 1
    flops, macs, params = get_model_profile(
        model=model,  # model
        input_shape=None,  # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
        args=[
            torch.randn([batch_size, 1, 1025, 282]),
            torch.randn([batch_size, 2, 1025, 282]),
            torch.randn([batch_size, 2, 1025, 282]),
        ],  # list of positional arguments to the model.
        kwargs={},  # dictionary of keyword arguments to the model.
        print_profile=True,  # prints the model graph with the measured profile attached to each module
        detailed=True,  # print the detailed profile
        module_depth=-1,  # depth into the nested modules, with -1 being the inner most modules
        top_modules=1,  # the number of top modules to print aggregated profile
        warm_up=1,  # the number of warm-ups before measuring the time of each module
        as_string=True,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None,
    )  # the list of modules to ignore in the profiling
