from asteroid.dsp.overlap_add import LambdaOverlapAdd

from . import (
    LambdaOverlapAdd_norm,
    LambdaOverlapAdd_Wav2Vec,
    LambdaOverlapAdd_Chunkwise_Wav2Vec,
    LambdaOverlapAdd_Chunkwise_SpectralFeatures,
)


def load_ola_func_with_args(args, model, device, meter):
    if args.use_overlapadd == "ola":
        continuous_nnet = LambdaOverlapAdd(
            nnet=model,
            n_src=2,
            window_size=int(args.seq_dur * args.sample_rate)
            if not args.ola_window_len
            else int(args.ola_window_len * args.sample_rate),
            hop_size=int(args.seq_dur * args.sample_rate // 4)
            if not args.ola_hop_len
            else int(args.ola_hop_len * args.sample_rate),
            window=None,
            reorder_chunks=args.reorder_chunks,
            enable_grad=False,
        ).to(device)
    elif args.use_overlapadd == "ola_norm":
        continuous_nnet = LambdaOverlapAdd_norm(
            nnet=model,
            n_src=2,
            window_size=int(args.seq_dur * args.sample_rate)
            if not args.ola_window_len
            else int(args.ola_window_len * args.sample_rate),
            hop_size=int(args.seq_dur * args.sample_rate // 4)
            if not args.ola_hop_len
            else int(args.ola_hop_len * args.sample_rate),
            window=None,
            reorder_chunks=args.reorder_chunks,
            enable_grad=False,
            target_lufs=-24,
            meter=meter,
            device=device,
        ).to(device)
    elif args.use_overlapadd == "w2v":
        continuous_nnet = LambdaOverlapAdd_Wav2Vec(
            nnet=model,
            n_src=2,
            window_size=int(args.seq_dur * args.sample_rate)
            if not args.ola_window_len
            else int(args.ola_window_len * args.sample_rate),
            hop_size=int(args.seq_dur * args.sample_rate // 4)
            if not args.ola_hop_len
            else int(args.ola_hop_len * args.sample_rate),
            window=None,
            reorder_chunks=args.reorder_chunks,
            enable_grad=False,
            device=device,
            sr=args.sample_rate,
            w2v_checkpoint_dir=args.w2v_ckpt_dir,
            w2v_nth_layer_output=args.w2v_nth_layer_output,
        ).to(device)
    elif args.use_overlapadd == "w2v_chunk":
        continuous_nnet = LambdaOverlapAdd_Chunkwise_Wav2Vec(
            nnet=model,
            n_src=2,
            window_size=args.nfft,
            hop_size=None,
            window=None,
            reorder_chunks=args.reorder_chunks,
            enable_grad=False,
            device=device,
            sr=args.sample_rate,
            vad_method=args.vad_method,
            w2v_checkpoint_dir=args.w2v_ckpt_dir,
            w2v_nth_layer_output=args.w2v_nth_layer_output,
        ).to(device)
    elif (
        args.use_overlapadd == "sf_chunk"
    ):  # chunk-wise processing based on spectral feature based reordering
        continuous_nnet = LambdaOverlapAdd_Chunkwise_SpectralFeatures(
            nnet=model,
            n_src=2,
            window_size=args.nfft,
            hop_size=None,
            window=None,
            reorder_chunks=args.reorder_chunks,
            enable_grad=False,
            device=device,
            sr=args.sample_rate,
            vad_method=args.vad_method,
            spectral_features=args.spectral_features,
        ).to(device)

    return continuous_nnet
