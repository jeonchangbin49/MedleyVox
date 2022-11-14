from asteroid_filterbanks import (
    make_enc_dec,
)
from asteroid.models.base_models import BaseEncoderMaskerDecoder
from asteroid.models import ConvTasNet
from asteroid.masknn import TDConvNet

from . import (
    BaseEncoderMaskerDecoder_mixture_consistency,
    BaseEncoderMaskerDecoder_mixture_consistency_super_resolution,
    SepFormer,
    TDConvNetpp_modified,
)
from .super_resolution_net import SFSRNet, SFSRNet_ConvNext


def load_model_with_args(args):
    if args.architecture == "conv_tasnet_stft":
        encoder, decoder = make_enc_dec(
            "torch_stft",
            n_filters=args.nfft,
            kernel_size=args.nfft,
            stride=args.nhop,
            sample_rate=args.sample_rate,
        )
        masker = TDConvNet(
            in_chan=encoder.n_feats_out,
            n_src=args.n_src,
            out_chan=None,
            n_blocks=args.n_blocks,
            n_repeats=args.n_repeats,
            bn_chan=args.bn_chan,
            hid_chan=args.hid_chan,
            skip_chan=args.skip_chan,
            # conv_kernel_size=conv_kernel_size,
            # norm_type=norm_type,
            mask_act=args.mask_act,
            # causal=causal,
        )

    elif args.architecture == "tdcnpp_stft":
        encoder, decoder = make_enc_dec(
            "torch_stft",
            n_filters=args.nfft,
            kernel_size=args.nfft,
            stride=args.nhop,
            sample_rate=args.sample_rate,
        )
        masker = TDConvNetpp_modified(
            in_chan=encoder.n_feats_out,
            n_src=args.n_src,
            out_chan=None,
            n_blocks=args.n_blocks,
            n_repeats=args.n_repeats,
            bn_chan=args.bn_chan,
            hid_chan=args.hid_chan,
            skip_chan=args.skip_chan,
            # conv_kernel_size=conv_kernel_size,
            # norm_type=norm_type,
            mask_act=args.mask_act,
            # causal=causal,
        )

    elif args.architecture == "conv_tasnet_learnable_basis":
        encoder, decoder = make_enc_dec(
            "free",
            n_filters=args.n_filter,
            kernel_size=args.n_kernel,
            stride=args.nhop,
            sample_rate=args.sample_rate,
        )
        masker = TDConvNet(
            in_chan=encoder.n_feats_out,
            n_src=args.n_src,
            out_chan=None,
            n_blocks=args.n_blocks,
            n_repeats=args.n_repeats,
            bn_chan=args.bn_chan,
            hid_chan=args.hid_chan,
            # skip_chan=skip_chan,
            # conv_kernel_size=conv_kernel_size,
            # norm_type=norm_type,
            mask_act=args.mask_act,
            # causal=causal,
        )

    elif args.architecture == "conv_tasnet_pretrained":

        model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")

    elif args.architecture == "sepformer_learnable_basis":
        encoder, decoder = make_enc_dec(
            "free",
            n_filters=args.n_filter,
            kernel_size=args.n_kernel,
            stride=args.nhop,
            sample_rate=args.sample_rate,
        )
        masker = SepFormer(
            in_chan=encoder.n_feats_out,
            n_src=args.n_src,
            n_blocks=args.n_blocks,
            n_repeats=args.n_repeats,
            ff_hid=args.hid_chan,
            # skip_chan=skip_chan,
            # conv_kernel_size=conv_kernel_size,
            # norm_type=norm_type,
            ff_activation=args.ff_activation,
            mask_act=args.mask_act,
            # causal=causal,
        )
    elif args.architecture == "sepformer_stft":
        encoder, decoder = make_enc_dec(
            "torch_stft",
            n_filters=args.nfft,
            kernel_size=args.nfft,
            stride=args.nhop,
            sample_rate=args.sample_rate,
        )
        masker = SepFormer(
            in_chan=encoder.n_feats_out,
            n_src=args.n_src,
            n_blocks=args.n_blocks,
            n_repeats=args.n_repeats,
            ff_hid=args.hid_chan,
            # skip_chan=skip_chan,
            # conv_kernel_size=conv_kernel_size,
            # norm_type=norm_type,
            ff_activation=args.ff_activation,
            mask_act=args.mask_act,
            # causal=causal,
        )

    if args.mixture_consistency == "mixture_consistency":
        # refer the details in "S.Wisdom, et al. 2018. DIFFERENTIABLE CONSISTENCY CONSTRAINTS FOR IMPROVED DEEP SPEECH ENHANCEMENT"
        # we performed several preliminary studies on masking strategy or residual calculation for consistent mixture, but this works best.
        model = BaseEncoderMaskerDecoder_mixture_consistency(
            encoder,
            masker,
            decoder,
            encoder_activation=args.encoder_activation,
        )
    elif args.mixture_consistency == "sfsrnet":
        if args.srnet == "orig":
            sr_net = SFSRNet(n_src=args.n_src, norm_type="gLN")
        elif args.srnet == "convnext":
            sr_net = SFSRNet_ConvNext(n_src=args.n_src, norm_type="gLN")
        model = BaseEncoderMaskerDecoder_mixture_consistency_super_resolution(
            encoder,
            masker,
            decoder,
            sr_net,
            window_size=args.nfft,
            above_freq=args.above_freq,
            sample_rate=args.sample_rate,
            encoder_activation=args.encoder_activation,
            db_normalize=args.db_normalize,
            sr_input_res=args.sr_input_res,
            sr_out_mix_consistency=args.sr_out_mix_consistency
            if hasattr(args, "sr_out_mix_consistency")
            else False,
        )
    else:
        model = BaseEncoderMaskerDecoder(
            encoder,
            masker,
            decoder,
            encoder_activation=args.encoder_activation,
        )

    return model
