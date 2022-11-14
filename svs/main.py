import os
import random
import argparse
import warnings

warnings.filterwarnings("ignore", message="There were no voiced segments found.")

import torch

from .train import train
from .utils import str2bool


def main():
    parser = argparse.ArgumentParser(description="Singing separation Trainer")

    # wandb params
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--entity", type=str, default="your_entity_id")
    parser.add_argument("--project", type=str, default="your_project_name")
    parser.add_argument("--sweep", type=str2bool, default=False)

    parser.add_argument(
        "--target",
        type=str,
        default="vocals",
        help="target source (will be passed to the dataset)",
    )
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
        help="network architecture",
    )
    parser.add_argument("--mask_act", type=str, default="linear", help="relu, linear")
    parser.add_argument(
        "--ff_activation", type=str, default="relu", help="relu, linear, gelu, etc."
    )
    parser.add_argument(
        "--encoder_activation", type=str, default=None, help="relu, linear"
    )
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
    parser.add_argument(
        "--sr_out_mix_consistency",
        type=str2bool,
        default=False,
        help="when using sfsrnet, apply mixture consistency constraint on srnet output?",
    )

    # Network parameters
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=6,
        help="Number of convolutional blocks in each repeat. Defaults to 6.",
    )
    parser.add_argument(
        "--n_repeats", type=int, default=4, help="Number of repeats. Defaults to 4."
    )
    parser.add_argument(
        "--bn_chan",
        type=int,
        default=256,
        help="Number of channels after the bottleneck.",
    )
    parser.add_argument(
        "--skip_chan",
        type=int,
        default=256,
        help="Number of channels of skip connection outputs.",
    )
    parser.add_argument(
        "--hid_chan",
        type=int,
        default=1024,
        help="Number of channels in the convolutional blocks.",
    )

    # Dataset paramaters
    parser.add_argument(
        "--dataset",
        type=str,
        default="singing_librispeech",
        choices=[
            "singing_librispeech",
            "multi_singing_librispeech",
        ],
        help="Name of the dataset. Duet for singing_librispeech, Main vs. rest for multi_singing_librispeech",
    )
    parser.add_argument(
        "--sing_sing_ratio",
        type=float,
        default=0.15,
        help="singing+singing train dataset portion",
    )
    parser.add_argument(
        "--sing_speech_ratio",
        type=float,
        default=0.15,
        help="singing+singing train dataset portion",
    )
    parser.add_argument(
        "--same_song_ratio",
        type=float,
        default=0.2,
        help="singing+singing train dataset portion",
    )
    parser.add_argument(
        "--same_singer_ratio",
        type=float,
        default=0.2,
        help="singing+singing train dataset portion",
    )
    parser.add_argument(
        "--same_speaker_ratio",
        type=float,
        default=0.15,
        help="singing+singing train dataset portion",
    )
    parser.add_argument(
        "--reduced_training_data_ratio",
        type=float,
        default=1.0,
        help="since sfsrnet took so long time to training, reduced the train data size. Different from --part_of_data",
    )

    # Data augmentation parameters
    parser.add_argument(
        "--unison_prob",
        type=float,
        default=0.3,
        help="unison augmentation probability. If 0., no augmentation",
    )
    parser.add_argument(
        "--pitch_formant_augment_prob",
        type=float,
        default=0.4,
        help="pitch shift + formant augmentation. If 0., no augmentation",
    )

    # Training dataset directories
    # 24k --train_root
    parser.add_argument(
        "--train_root",
        nargs="+",
        default=[
            "/path/to/data/24k/CSD",
            "/path/to/data/24k/NUS",
            "/path/to/data/24k/TONAS",
            "/path/to/data/24k/VocalSet",
            "/path/to/data/24k/jsut-song_ver1",
            "/path/to/data/24k/jvs_music_ver1",
            "/path/to/data/24k/kiritan_revised",
            "/path/to/data/24k/vocadito",
            "/path/to/data/24k/musdb_a_train",
            "/path/to/data/24k/OpenSinger",
            "/path/to/data/24k/medleyDB_v1_in_musdb",
            # '/path/to/data/24k/medleyDB_v1_rest',
            # '/path/to/data/24k/medleyDB_v2_rest',
            "/path/to/data/24k/k_multisinger",
            "/path/to/data/24k/k_multitimbre",
        ],
        help="root path list of dataset",
    )

    # 24k --speech_train_root
    parser.add_argument(
        "--speech_train_root",
        nargs="+",
        default=[
            "/path/to/data/24k/LibriSpeech_train-clean-360",
            "/path/to/data/24k/LibriSpeech_train-clean-100",
        ],
        help="root path list of dataset",
    )

    # 24k --same_song_dict_path
    parser.add_argument(
        "--same_song_dict_path",
        nargs="+",
        action="append",
        default=[
            [
                "/path/to/data/24k/k_multisinger",
                "./svs/preprocess/make_same_song_dict/same_song_k_multisinger.json",
                "k_multisinger",
            ]
        ],
        help="For making the dataloader that outputs source1 and source2 from SAME song. list of [[data_root,data_dict_path, data_name], ...]",
    )

    # 24k --same_singer_dict_path
    parser.add_argument(
        "--same_singer_dict_path",
        nargs="+",
        action="append",
        default=[
            [
                "/path/to/data/24k/OpenSinger",
                "./svs/preprocess/make_same_singer_dict/same_singer_OpenSinger.json",
                "OpenSinger",
            ],
            [
                "/path/to/data/24k/k_multisinger",
                "./svs/preprocess/make_same_singer_dict/same_singer_k_multisinger.json",
                "k_multisinger",
            ],
            [
                "/path/to/data/24k/CSD",
                "./svs/preprocess/make_same_singer_dict/same_singer_CSD.json",
                "CSD",
            ],
            [
                "/path/to/data/24k/jsut-song_ver1",
                "./svs/preprocess/make_same_singer_dict/same_singer_jsut-song_ver1.json",
                "jsut-song_ver1",
            ],
            [
                "/path/to/data/24k/jvs_music_ver1",
                "./svs/preprocess/make_same_singer_dict/same_singer_jvs_music_ver1.json",
                "jvs_music_ver1",
            ],
            [
                "/path/to/data/24k/k_multitimbre",
                "./svs/preprocess/make_same_singer_dict/same_singer_k_multitimbre.json",
                "k_multitimbre",
            ],
            [
                "/path/to/data/24k/kiritan_revised",
                "./svs/preprocess/make_same_singer_dict/same_singer_kiritan.json",
                "kiritan",
            ],
            [
                "/path/to/data/24k/musdb_a_train",
                "./svs/preprocess/make_same_singer_dict/same_singer_musdb_a_train.json",
                "musdb_a_train",
            ],
            [
                "/path/to/data/24k/NUS",
                "./svs/preprocess/make_same_singer_dict/same_singer_NUS.json",
                "NUS",
            ],
            [
                "/path/to/data/24k/VocalSet",
                "./svs/preprocess/make_same_singer_dict/same_singer_VocalSet.json",
                "VocalSet",
            ],
        ],
        help="For making the dataloader that outputs source1 and source2 from SAME singer. list of [[data_root,data_dict_path, data_name], ...]",
    )

    # 24k --same_speaker_dict_path
    parser.add_argument(
        "--same_speaker_dict_path",
        nargs="+",
        action="append",
        default=[
            [
                "/path/to/data/24k/LibriSpeech_train-clean-100",
                "./svs/preprocess/make_same_speaker_dict/same_singer_LibriSpeech_train-clean-100.json",
                "LibriSpeech_train-clean-100",
            ],
            [
                "/path/to/data/24k/LibriSpeech_train-clean-360",
                "./svs/preprocess/make_same_speaker_dict/same_singer_LibriSpeech_train-clean-360.json",
                "LibriSpeech_train-clean-360",
            ],
        ],
        help="For making the dataloader that outputs source1 and source2 from SAME speaker. list of [[data_root,data_dict_path, data_name], ...]",
    )

    # Validation dataset directories
    # 24k --valid_root
    parser.add_argument(
        "--valid_root",
        nargs="+",
        action="append",
        default=[
            [
                "/path/to/data/24k/musdb_a_test",
                "/path/to/data/24k/musdb_a_test",
                "./svs/preprocess/make_validation_dict/for_2_srcs/valid_regions_dict_singing_singing.json",
                "sing_sing_diff",
            ],
            [
                "/path/to/data/24k/musdb_a_test",
                "/path/to/data/24k/musdb_a_test",
                "./svs/preprocess/make_validation_dict/for_2_srcs/valid_regions_dict_singing_unison.json",
                "sing_sing_unison",
            ],
            [
                "/path/to/data/24k/musdb_a_test",
                "/path/to/data/24k/musdb_a_test",
                "./svs/preprocess/make_validation_dict/for_2_srcs/valid_regions_dict_singing_singing_same_singer.json",
                "sing_sing_same_singer",
            ],
            [
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "./svs/preprocess/make_validation_dict/for_2_srcs/valid_regions_dict_speech_speech.json",
                "speech_speech_diff",
            ],
            [
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "./svs/preprocess/make_validation_dict/for_2_srcs/valid_regions_dict_speech_unison.json",
                "speech_speech_unison",
            ],
            [
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "./svs/preprocess/make_validation_dict/for_2_srcs/valid_regions_dict_speech_speech_same_speaker.json",
                "speech_speech_same_speaker",
            ],
            [
                "/path/to/data/24k/musdb_a_test",
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "./svs/preprocess/make_validation_dict/for_2_srcs/valid_regions_dict_singing_speech.json",
                "singing_speech",
            ],
        ],
        help="root path list of dataset. list of [source1 data_dir, source2 data_dir, data_region_info_dict]",
    )

    # OR-PIT dataset directories
    parser.add_argument(
        "--valid_root_orpit",
        nargs="+",
        action="append",
        default=[
            [
                "/path/to/data/24k/musdb_a_test",
                "/path/to/data/24k/musdb_a_test",
                "./svs/preprocess/make_validation_dict/for_n_srcs/valid_regions_dict_singing_singing_n_srcs.json",
                "sing_sing_diff",
            ],
            [
                "/path/to/data/24k/musdb_a_test",
                "/path/to/data/24k/musdb_a_test",
                "./svs/preprocess/make_validation_dict/for_n_srcs/valid_regions_dict_singing_unison_n_srcs.json",
                "sing_sing_unison",
            ],
            [
                "/path/to/data/24k/musdb_a_test",
                "/path/to/data/24k/musdb_a_test",
                "./svs/preprocess/make_validation_dict/for_n_srcs/valid_regions_dict_singing_singing_same_singer_n_srcs.json",
                "sing_sing_same_singer",
            ],
            [
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "./svs/preprocess/make_validation_dict/for_n_srcs/valid_regions_dict_speech_speech_n_srcs.json",
                "speech_speech_diff",
            ],
            [
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "./svs/preprocess/make_validation_dict/for_n_srcs/valid_regions_dict_speech_unison_n_srcs.json",
                "speech_speech_unison",
            ],
            [
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "./svs/preprocess/make_validation_dict/for_n_srcs/valid_regions_dict_speech_speech_same_speaker_n_srcs.json",
                "speech_speech_same_speaker",
            ],
            [
                "/path/to/data/24k/musdb_a_test",
                "/path/to/data/24k/LibriSpeech_dev-clean",
                "./svs/preprocess/make_validation_dict/for_n_srcs/valid_regions_dict_singing_speech_n_srcs.json",
                "singing_speech",
            ],
        ],
        help="root path list of dataset. list of [source1 data_dir, source2 data_dir, data_region_info_dict]",
    )
    # 24k --song_length_dict_path
    parser.add_argument(
        "--song_length_dict_path",
        type=str,
        default="./svs/preprocess/song_length_dict_24k.json",
        help="path of json file that contains the lengths of data",
    )

    parser.add_argument(
        "--min_n_src",
        type=int,
        default=2,
        help="minimum number of sources in mixture during OR-PIT trianing",
    )
    parser.add_argument(
        "--max_n_src",
        type=int,
        default=4,
        help="maximum number of sources in mixture during OR-PIT trianing",
    )

    parser.add_argument(
        "--valid_regions_dict_path",
        type=str,
        default="./svs/preprocess/valid_regions_dict_singing_singing.json",
        help="path of json file that contains the lengths of data",
    )

    parser.add_argument(
        "--output_directory", type=str, default="/path/to/results/singing_sep"
    )
    parser.add_argument("--exp_name", type=str)
    parser.add_argument(
        "--part_of_data",
        type=float,
        default=None,
        help="to check the effect of data amount",
    )

    parser.add_argument("--ema", type=str2bool, default=True, help="use model ema?")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="which optimizer do you want to use?",
    )

    # resume training or continue.
    parser.add_argument("--resume", type=str, help="path to checkpoint folder")
    parser.add_argument(
        "--continual_train",
        type=str2bool,
        default=False,
        help="continue training from the pre-trained checkpoints",
    )
    parser.add_argument(
        "--load_ema_online_model",
        type=str2bool,
        default=False,
        help="continue training from the online model from the ema pre-trained checkpoints. To use this, make sure --ema=False",
    )
    parser.add_argument(
        "--start_from_best",
        type=str2bool,
        default=False,
        help="when use --continual_train, do you want to start from previous best-performed weight?",
    )

    # Trainig Parameters
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="learning rate, defaults to 2e-4"
    )
    parser.add_argument("--beta1", type=float, default=0.5, help="adam optimizer beta1")
    parser.add_argument("--beta2", type=float, default=0.9, help="adam optimizer beta2")
    parser.add_argument(
        "--eps", type=float, default=1e-8, help="optimizer eps parameter"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="maximum number of epochs to train (my default: 80)",
    )
    parser.add_argument(
        "--lr_decay_patience",
        type=int,
        default=20,
        help="lr decay patience for plateau scheduler (my default : 25)",
    )
    parser.add_argument(
        "--lr_decay_gamma",
        type=float,
        default=0.5,
        help="gamma of learning rate scheduler decay",
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="step_lr", help="step_lr, cos_warmup"
    )

    parser.add_argument(
        "--train_loss_func",
        nargs="+",
        default=["pit_snr", "multi_spectral_l1"],
        # default=["pit_snr", "multi_spectral_l1", "snr"], # when training iSRNet, use this.
        help="mse, L1, pit_si_sdr, pit_sd_sdr, pit_sdr, multi_spectral, multi_spectral_l1",
    )
    parser.add_argument(
        "--valid_loss_func",
        nargs="+",
        default=["pit_si_sdr"],
        help="keep this unchanged for validation loss check.",
    )
    parser.add_argument(
        "--above_freq",
        type=float,
        default=300.0,
        help="if you want to calc spectral loss only above --above_freq. Applicable when using multi_spectral_l1_above_freq in train. Also applicable when using iSRNet Heuristic calculation.",
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
        default=24000,
        help="Sequence duration in seconds"
        "value of <=0.0 will use full/variable length",
    )

    parser.add_argument(
        "--seq_dur",
        type=float,
        default=3.0,
        help="Sequence duration in seconds"
        "value of <=0.0 will use full/variable length",
    )
    parser.add_argument(
        "--n_src", type=int, default=2, help="number of estimating sources"
    )
    parser.add_argument(
        "--nfft", type=int, default=2048, help="STFT fft size and window size"
    )
    parser.add_argument("--nhop", type=int, default=512, help="STFT hop size")
    parser.add_argument(
        "--n_filter", type=int, default=512, help="learnable basis filter size"
    )
    parser.add_argument(
        "--n_kernel", type=int, default=512, help="learnable basis kernel size"
    )

    parser.add_argument(
        "--nb_workers", type=int, default=4, help="Number of workers for dataloader."
    )

    # Misc Parameters
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    parser.add_argument("--port", default=None, type=str, help="port")

    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=4, type=int)
    parser.add_argument("--n_nodes", default=1, type=int)

    args, _ = parser.parse_known_args()

    args.output = f"{args.output_directory}/checkpoint/{args.exp_name}"

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(random.randint(0, 1800))

    os.makedirs(args.output, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(args)
    train(args)


if __name__ == "__main__":
    main()
