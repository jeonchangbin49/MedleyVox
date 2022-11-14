import torch
import torch.nn.functional as F
from asteroid.dsp.overlap_add import LambdaOverlapAdd
from asteroid.losses.pit_wrapper import PITLossWrapper

from .wav2vec_feature_extractor import Wav2Vec2FeatureExtractor
from ..utils.logging import AverageMeter


class LambdaOverlapAdd_Wav2Vec(LambdaOverlapAdd):
    """
    Basically, this is same as the original LambdaOverlapAdd in asteroid.
    But, similarity of overlapped regions is calculated based on w2v features, not just a waveform L1 difference in asteroid.
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
        w2v_checkpoint_dir=None,
        w2v_nth_layer_output=[0],
    ):
        super().__init__(
            nnet, n_src, window_size, hop_size, window, reorder_chunks, enable_grad
        )
        self.nnet = self.nnet.to(device)
        self.device = device
        self.sr = sr
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
        # Overlap and add:
        # [batch, chans, n_frames] -> [batch, chans, win_size, n_chunks]
        # if chans=1 then [batch, n_frames] -> [batch, win_size, n_chunks]
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.window_size, 1),
            padding=(self.window_size, 0),
            stride=(self.hop_size, 1),
        )
        out = []
        n_chunks = unfolded.shape[-1]
        for frame_idx in range(n_chunks):  # for loop to spare memory
            frame = self.nnet(unfolded[..., frame_idx])
            # user must handle multichannel by reshaping to batch
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
                    )  # we want wav2vec2.0's 0th layer output
                w2v_output_list = torch.cat(
                    w2v_output_list, dim=1
                )  # [batch, n_src, feature_dim]
                self.w2v_out_avg.update(w2v_output_list)

            frame = frame.reshape(batch * n_src, -1)

            if frame_idx != 0 and self.reorder_chunks:
                # we determine best perm based on xcorr with previous sources
                frame, w2v_out = self._reorder_sources_based_on_w2v(
                    frame,
                    out[-1],
                    self.w2v_out_avg.avg,
                    n_src,
                    self.window_size,
                    self.hop_size,
                )
                self.w2v_out_avg.update(w2v_out)
            if self.use_window:
                frame = frame * self.window
            else:
                frame = frame / (self.window_size / self.hop_size)
            out.append(frame)

        out = torch.stack(out).reshape(n_chunks, batch * n_src, self.window_size)
        out = out.permute(1, 2, 0)

        out = torch.nn.functional.fold(
            out,
            (n_frames, 1),
            kernel_size=(self.window_size, 1),
            padding=(self.window_size, 0),
            stride=(self.hop_size, 1),
        )
        return out.squeeze(-1).reshape(batch, n_src, -1)

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

    def _reorder_sources_based_on_w2v(
        self,
        current: torch.FloatTensor,
        previous: torch.FloatTensor,
        previous_w2v: torch.FloatTensor,
        n_src: int,
        window_size: int,
        hop_size: int,
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
            n_src (:class:`int`): number of sources.
            window_size (:class:`int`): window_size, equal to last dimension of
                                        both current and previous.
            hop_size (:class:`int`): hop_size between current and previous tensors.

        """
        batch, frames = current.size()
        current = current.reshape(-1, n_src, frames)
        previous = previous.reshape(-1, n_src, frames)

        overlap_f = window_size - hop_size

        def reorder_func_w2v(x):
            x = x[..., :overlap_f]

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
            current.reshape(batch, frames),
            current_w2v,
        )  # this 'batch' is not real batch, it's actually batch * n_src


class PITLossWrapper_Out_BatchIndices(PITLossWrapper):
    def __init__(self, loss_func, pit_from="pw_mtx", perm_reduce=None):
        super().__init__(loss_func, pit_from, perm_reduce)

    def forward(
        self, est_targets, targets, return_est=True, reduce_kwargs=None, **kwargs
    ):
        n_src = targets.shape[1]
        assert n_src < 10, f"Expected source axis along dim 1, found {n_src}"
        if self.pit_from == "pw_mtx":  # Only applicable with "pw_mtx"
            # Loss function already returns pairwise losses
            pw_losses, w2v_output_list = self.loss_func(est_targets, **kwargs)
        assert pw_losses.ndim == 3, (
            "Something went wrong with the loss " "function, please read the docs."
        )
        assert (
            pw_losses.shape[0] == targets.shape[0]
        ), "PIT loss needs same batch dim as input"

        reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()
        min_loss, batch_indices = self.find_best_perm(
            pw_losses, perm_reduce=self.perm_reduce, **reduce_kwargs
        )
        mean_loss = torch.mean(min_loss)
        if not return_est:
            return mean_loss
        reordered = self.reorder_source(est_targets, batch_indices)
        reordered_w2v_output = self.reorder_source(w2v_output_list, batch_indices)

        # Output batch indices added (different with original asteroid code)
        return mean_loss, reordered, reordered_w2v_output
