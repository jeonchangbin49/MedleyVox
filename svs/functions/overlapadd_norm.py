# modify asteroid's LambdaOverlapAdd to normalize the input
import torch
from asteroid.dsp.overlap_add import LambdaOverlapAdd, _reorder_sources

from ..utils import loudnorm


class LambdaOverlapAdd_norm(LambdaOverlapAdd):
    def __init__(
        self,
        nnet,
        n_src,
        window_size,
        hop_size=None,
        window="hanning",
        reorder_chunks=True,
        enable_grad=False,
        target_lufs=-24,
        meter=None,
        device="cpu",
    ):
        super().__init__(
            nnet, n_src, window_size, hop_size, window, reorder_chunks, enable_grad
        )
        self.nnet = self.nnet.to(device)
        self.target_lufs = target_lufs
        self.meter = meter  # Loudness meter
        self.device = device

    def ola_forward(self, x):
        """Heart of the class: segment signal, apply func, combine with OLA."""

        assert x.ndim == 3

        batch, channels, n_frames = x.size()
        # Overlap and add:
        # [batch, chans, n_frames] -> [batch, chans, win_size, n_chunks]
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.window_size, 1),
            padding=(self.window_size, 0),
            stride=(self.hop_size, 1),
        )

        out = []
        n_chunks = unfolded.shape[-1]
        for frame_idx in range(n_chunks):  # for loop to spare memory
            unfolded_temp = unfolded[..., frame_idx].cpu().numpy()
            unfolded_temp, adjusted_gain = loudnorm(
                unfolded_temp.squeeze(0).T, self.target_lufs, self.meter
            )
            unfolded_temp = torch.as_tensor(unfolded_temp, dtype=torch.float32)
            unfolded_temp = unfolded_temp.to(self.device)
            unfolded_temp = unfolded_temp.unsqueeze(0)

            frame = self.nnet(unfolded_temp)
            # user must handle multichannel by reshaping to batch
            if frame_idx == 0:
                assert frame.ndim == 3, "nnet should return (batch, n_src, time)"
                if self.n_src is not None:
                    assert (
                        frame.shape[1] == self.n_src
                    ), "nnet should return (batch, n_src, time)"
                n_src = frame.shape[1]
            frame = frame.reshape(batch * n_src, -1)

            if frame_idx != 0 and self.reorder_chunks:
                # we determine best perm based on xcorr with previous sources
                frame = _reorder_sources(
                    frame, out[-1], n_src, self.window_size, self.hop_size
                )

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
