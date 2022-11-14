import torch
import torch.nn.functional as F
from asteroid.dsp.overlap_add import LambdaOverlapAdd
import librosa


from ..utils.logging import AverageMeter
from .silence_split import magspec_vad, webrtc_vad
from .overlapadd_w2v import PITLossWrapper_Out_BatchIndices


class LambdaOverlapAdd_Chunkwise_SpectralFeatures(LambdaOverlapAdd):
    """
    Code for Chunk-wise processing, assignment is perfomed by Spectral features (here, we used mfcc or spectral centroid)
    """

    def __init__(
        self,
        nnet,
        n_src,
        window_size,
        hop_size=None,
        window="hanning",
        reorder_chunks=True,
        enable_grad=False,
        device="cpu",
        sr=24000,
        vad_method="spec",
        spectral_features="mfcc",
    ):
        super().__init__(
            nnet, n_src, window_size, hop_size, window, reorder_chunks, enable_grad
        )
        self.nnet = self.nnet.to(device)
        self.device = device
        self.sr = sr
        self.vad_method = vad_method
        self.spectral_features = spectral_features

    def ola_forward(self, x):
        """Heart of the class: segment signal, apply func, combine with OLA."""
        self.sc_avg = AverageMeter()  # to cumulate previous spectral centroids

        assert x.ndim == 3

        batch, channels, n_frames = x.size()

        if self.vad_method == "spec":
            starts, ends = magspec_vad(
                x.cpu().numpy()[0, 0, :],
                n_fft=self.window_size,
                hop_length=self.hop_size,
            )
        elif self.vad_method == "webrtc":
            starts, ends = webrtc_vad(
                x.cpu().numpy()[0, 0, :], self.sr, vad_mode=3, frame_size=0.03
            )
        # First, make the output tensor. divide by n_src will make sum of the output be consistent with input
        # except the regions where voice activity detected.
        out = (x / self.n_src).repeat(1, self.n_src, 1)  # [batch, n_src, n_frames]
        assert len(starts) == len(ends)

        for frame_idx in range(len(starts)):  # for loop to spare memory
            frame_length = ends[frame_idx] - starts[frame_idx]
            if (
                frame_length <= self.window_size // 2
            ):  # if input frames are too short, an error occurs.
                pad_each_side = int((self.window_size // 2 - frame_length) / 2) + 1
                segment = F.pad(
                    x[..., starts[frame_idx] : ends[frame_idx]],
                    (pad_each_side, pad_each_side),
                )
                frame = self.nnet(segment)
            else:
                segment = x[..., starts[frame_idx] : ends[frame_idx]]
                frame = self.nnet(segment)
            if frame_idx == 0:
                assert frame.ndim == 3, "nnet should return (batch, n_src, time)"
                if self.n_src is not None:
                    assert (
                        frame.shape[1] == self.n_src
                    ), "nnet should return (batch, n_src, time)"
                n_src = frame.shape[1]
                sf_output_list = []
                for src in range(n_src):
                    if self.spectral_features == "mfcc":
                        spec_feat_output = torch.as_tensor(
                            librosa.feature.mfcc(
                                y=frame[0, src, :].cpu().numpy(),
                                sr=self.sr,
                                n_mfcc=20,
                                n_fft=self.window_size,
                                hop_length=self.hop_size,
                            )[1:, :]
                            .mean(1, keepdims=True)
                            .T,
                            device=self.device,
                        ).unsqueeze(0)
                    elif self.spectral_features == "spectral_centroid":
                        spec_feat_output = torch.as_tensor(
                            librosa.feature.spectral_centroid(
                                y=frame[0, src, :].cpu().numpy(),
                                sr=self.sr,
                                n_fft=self.window_size,
                                hop_length=self.hop_size,
                            ).mean(1, keepdims=True),
                            device=self.device,
                        ).unsqueeze(0)

                    sf_output_list.append(spec_feat_output)
                sf_output_list = torch.cat(
                    sf_output_list, dim=1
                )  # [batch, n_src, feature_dim]
                self.sc_avg.update(sf_output_list)

            if frame_idx != 0 and self.reorder_chunks:
                # we determine best perm based on xcorr with previous sources
                frame, sc_out = self._reorder_sources_with_sf_and_non_overlapped_seg(
                    frame,
                    out[..., starts[frame_idx - 1] : ends[frame_idx - 1]],
                    self.sc_avg.avg,
                    n_src,
                )
                self.sc_avg.update(sc_out)
            if frame_length <= self.window_size // 2:
                frame = frame[..., pad_each_side:-pad_each_side]

            out[..., starts[frame_idx] : ends[frame_idx]] = frame

        return out

    def forward(self, x):
        """Forward module: segment signal, apply func, combine with OLA.

        Args:
            x (:class:`torch.Tensor`): waveform signal of shape (batch, 1, time).

        Returns:
            :class:`torch.Tensor`: The output of the lambda OLA.
        """
        # Here we can do the reshaping
        with torch.autograd.set_grad_enabled(self.enable_grad):
            olad = self.ola_forward(x)
            return olad

    def _reorder_sources_with_sf_and_non_overlapped_seg(
        self,
        current: torch.FloatTensor,
        previous: torch.FloatTensor,
        previous_sf: torch.FloatTensor,
        n_src: int,
    ):
        """
        Reorder sources in current chunk to maximize correlation with previous chunk.
        Used for Continuous Source Separation. Wav2Vec2.0-based correlation is used
        for reordering.

        Args:
            current (:class:`torch.Tensor`): current chunk, tensor
                                            of shape (batch, n_src, window_size)
            previous (:class:`torch.Tensor`): previous chunk, tensor
                                            of shape (batch, n_src, window_size)
            n_src (:class:`int`): number of sources.
            window_size (:class:`int`): window_size, equal to last dimension of
                                        both current and previous.
            hop_size (:class:`int`): hop_size between current and previous tensors.

        """
        # batch, frames = current.size()
        batch, n_src, frames = current.size()

        def reorder_func_sf(x):

            sf_output_list = []
            for src in range(n_src):
                if self.spectral_features == "mfcc":
                    spec_feat_output = torch.as_tensor(
                        librosa.feature.mfcc(
                            y=x[0, src, :].cpu().numpy(),
                            sr=self.sr,
                            n_mfcc=20,
                            n_fft=self.window_size,
                            hop_length=self.hop_size,
                        )[1:, :]
                        .mean(1, keepdims=True)
                        .T,
                        device=self.device,
                    ).unsqueeze(0)
                elif self.spectral_features == "spectral_centroid":
                    spec_feat_output = torch.as_tensor(
                        librosa.feature.spectral_centroid(
                            y=x[0, src, :].cpu().numpy(),
                            sr=self.sr,
                            n_fft=self.window_size,
                            hop_length=self.hop_size,
                        ).mean(1, keepdims=True),
                        device=self.device,
                    ).unsqueeze(0)

                sf_output_list.append(spec_feat_output)
            sf_output_list = torch.cat(
                sf_output_list, dim=1
            )  # [batch, n_src, feature_dim]
            return (
                -F.cosine_similarity(
                    sf_output_list.unsqueeze(1), previous_sf.unsqueeze(2), dim=-1
                ),
                sf_output_list,
            )

        # We maximize correlation-like between previous and current.
        pit = PITLossWrapper_Out_BatchIndices(
            reorder_func_sf
        )  # So, reorder_func is a loss_function in PITLossWrapper

        _, current, current_sf = pit(current, previous)
        return (
            current,
            current_sf,
        )
