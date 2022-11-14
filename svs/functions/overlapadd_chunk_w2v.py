import torch
import torch.nn.functional as F
from asteroid.dsp.overlap_add import LambdaOverlapAdd

from .wav2vec_feature_extractor import Wav2Vec2FeatureExtractor
from ..utils.logging import AverageMeter
from .silence_split import magspec_vad, webrtc_vad
from .overlapadd_w2v import PITLossWrapper_Out_BatchIndices


class LambdaOverlapAdd_Chunkwise_Wav2Vec(LambdaOverlapAdd):
    """
    Code for Chunk-wise processing, assignment is perfomed by wav2vec2.0-xlsr features.
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
        w2v_checkpoint_dir=None,
        w2v_nth_layer_output=[0],
    ):
        super().__init__(
            nnet, n_src, window_size, hop_size, window, reorder_chunks, enable_grad
        )
        self.nnet = self.nnet.to(device)
        self.device = device
        self.sr = sr
        self.vad_method = vad_method
        self.w2v_model = Wav2Vec2FeatureExtractor(
            w2v_checkpoint_dir,
            sr,
            model_version="xlsr",
            using_feature="c",
            extraction_method="none",
            device=device,
        )
        self.w2v_nth_layer_output = w2v_nth_layer_output

    def ola_forward(self, x):
        """Heart of the class: segment signal, apply func, combine with OLA."""
        self.w2v_out_avg = AverageMeter()  # to cumulate previous wav2vec output

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
                w2v_output_list = []
                for src in range(n_src):
                    w2v_output_list.append(
                        self.w2v_model(
                            frame[:, src, :], nth_layer_output=self.w2v_nth_layer_output
                        ).mean(1, keepdim=True)
                    )
                w2v_output_list = torch.cat(
                    w2v_output_list, dim=1
                )  # [batch, n_src, feature_dim]
                self.w2v_out_avg.update(w2v_output_list)

            if frame_idx != 0 and self.reorder_chunks:
                # we determine best perm based on xcorr with previous sources
                frame, w2v_out = self._reorder_sources_with_w2v_and_non_overlapped_seg(
                    frame,
                    out[..., starts[frame_idx - 1] : ends[frame_idx - 1]],
                    self.w2v_out_avg.avg,
                    n_src,
                )
                self.w2v_out_avg.update(w2v_out)
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

    def _reorder_sources_with_w2v_and_non_overlapped_seg(
        self,
        current: torch.FloatTensor,
        previous: torch.FloatTensor,
        previous_w2v: torch.FloatTensor,
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
            previous_w2v (:class:'torch.Tensor'): previous w2v-chunk, tensor
        """
        # batch, frames = current.size()
        batch, n_src, frames = current.size()

        def reorder_func_w2v(x):

            w2v_output_list = []
            for src in range(n_src):
                w2v_output_list.append(
                    self.w2v_model(
                        x[:, src, :], nth_layer_output=self.w2v_nth_layer_output
                    ).mean(1, keepdim=True)
                )  # we want wav2vec2.0's 0th layer output
            w2v_output_list = torch.cat(
                w2v_output_list, dim=1
            )  # [batch, n_src, feature_dim]
            return (
                -F.cosine_similarity(
                    w2v_output_list.unsqueeze(1), previous_w2v.unsqueeze(2), dim=-1
                ),
                w2v_output_list,
            )

        # We maximize correlation-like between previous and current.
        pit = PITLossWrapper_Out_BatchIndices(
            reorder_func_w2v
        )  # So, reorder_func is a loss_function in PITLossWrapper

        _, current, current_w2v = pit(current, previous)
        return (
            current,
            current_w2v,
        )
