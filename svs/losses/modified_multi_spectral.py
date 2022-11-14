import torch
import torch.nn as nn
import numpy as np

from asteroid_filterbanks.transforms import mag
from asteroid.losses import SingleSrcMultiScaleSpectral


class SingleSrcMultiScaleSpectral_modified(SingleSrcMultiScaleSpectral):
    def __init__(self, n_filters=None, windows_size=None, hops_size=None, alpha=1.0):
        super().__init__(n_filters, windows_size, hops_size, alpha)

    def forward(self, est_target, target):
        batch_size = est_target.shape[0]
        est_target = est_target.unsqueeze(1)
        target = target.unsqueeze(1)

        loss = torch.zeros(batch_size, device=est_target.device)
        for encoder in self.encoders:
            loss += self.compute_spectral_loss(encoder, est_target, target)

        return loss.mean()

    def compute_spectral_loss(self, encoder, est_target, target, EPS=1e-8):
        batch_size = est_target.shape[0]
        spect_est_target = mag(encoder(est_target)).view(batch_size, -1)
        spect_target = mag(encoder(target)).view(batch_size, -1)
        linear_loss = nn.functional.l1_loss(spect_est_target, spect_target)
        log_loss = nn.functional.l1_loss(spect_est_target, spect_target)
        return linear_loss + self.alpha * log_loss


class SingleSrcMultiScaleSpectral_TRUnet(SingleSrcMultiScaleSpectral):
    def __init__(
        self,
        n_filters=[2048, 1024, 512],
        windows_size=[2048, 1024, 512],
        hops_size=[512, 256, 128],
        alpha=0.3,
        loss_scale=1.0,
        log_scale=False,
    ):
        super().__init__(n_filters, windows_size, hops_size, alpha)
        self.loss_scale = loss_scale
        self.log_scale = log_scale

    def forward(self, est_target, target):
        batch_size = est_target.shape[0]

        loss = torch.zeros(batch_size, device=est_target.device)
        for encoder in self.encoders:
            loss += self.compute_spectral_loss(encoder, est_target, target)

        if self.log_scale:
            return torch.log(loss.mean() * self.loss_scale + 1e-6)
        else:
            return loss.mean() * self.loss_scale

    def compute_spectral_loss(self, encoder, est_target, target, EPS=1e-8):
        batch_size = est_target.shape[0]

        spect_est_target = (
            mag(encoder(est_target)) ** self.alpha
        )  # ([160, 2, 1025, 90])
        spect_est_target = spect_est_target.view(batch_size, -1)
        spect_target = mag(encoder(target)) ** self.alpha
        spect_target = spect_target.view(batch_size, -1)

        loss = torch.norm(spect_target - spect_est_target)  # Frobenius Norm
        return loss


class SingleSrcMultiScaleSpectral_TRUNet_freq(SingleSrcMultiScaleSpectral):
    """
    Implementation of multi-resolution STFT loss in
    Real-time Denoising and Dereverberation with Tiny Recurrent U-Net (TRUNet)
    Hyeong-Seok Choi, et al. 2021 ICASSP.
    https://arxiv.org/abs/2102.03207
    """

    def __init__(
        self,
        n_filters=[2048, 1024, 512],
        windows_size=[2048, 1024, 512],
        hops_size=[512, 256, 128],
        alpha=0.3,
        loss_scale=1.0,
        sample_rate=24000,
        above_freq=1000,
        log_scale=False,
    ):
        super().__init__(n_filters, windows_size, hops_size, alpha)
        self.loss_scale = loss_scale
        self.over_freq_indices = []
        for each_window_size in windows_size:
            total_n_bins = int(1 + each_window_size / 2)
            stft_bins_freqs = (
                np.arange(0, total_n_bins) * sample_rate / each_window_size
            )  # make the array that contains each frequencis
            over_freq_index = np.where(stft_bins_freqs >= above_freq)[0][
                0
            ]  # to find where the starting frequency bin that exceeds 'above_freq'
            self.over_freq_indices.append(over_freq_index)
        self.log_scale = log_scale

    def forward(self, est_target, target):
        batch_size = est_target.shape[0]

        loss = torch.zeros(batch_size, device=est_target.device)
        for idx, encoder in enumerate(self.encoders):
            loss += self.compute_spectral_loss(
                encoder, est_target, target, self.over_freq_indices[idx]
            )

        if self.log_scale:
            return torch.log(loss.mean() * self.loss_scale + 1e-6)
        else:
            return loss.mean() * self.loss_scale

    def compute_spectral_loss(
        self, encoder, est_target, target, over_freq_index, EPS=1e-8
    ):
        batch_size = est_target.shape[0]

        # original spect_est_target size == ([160, 2, 1025, 90])
        spect_est_target = (
            mag(encoder(est_target))[..., over_freq_index:, :] ** self.alpha
        )
        spect_est_target = spect_est_target.view(batch_size, -1)
        spect_target = mag(encoder(target))[..., over_freq_index:, :] ** self.alpha
        spect_target = spect_target.view(batch_size, -1)

        loss = torch.norm(spect_target - spect_est_target)  # Frobenius Norm
        return loss


class SingleSrcMultiScaleSpectral_L1(SingleSrcMultiScaleSpectral):
    def __init__(
        self,
        n_filters=[2048, 1024, 512],
        windows_size=[2048, 1024, 512],
        hops_size=[512, 256, 128],
        alpha=None,
        loss_scale=1.0,
        log_scale=False,
    ):
        super().__init__(n_filters, windows_size, hops_size, alpha)
        # Compute L1 loss between targets and sources on each real and imaginary values
        self.loss_scale = loss_scale
        self.log_scale = log_scale

    def forward(self, est_target, target):
        est_target = est_target.unsqueeze(1)
        target = target.unsqueeze(1)

        loss = torch.zeros(1, device=est_target.device)
        for encoder in self.encoders:
            loss += self.compute_spectral_loss(encoder, est_target, target)

        if self.log_scale:
            return torch.log(loss.mean() + 1e-6)
        else:
            return loss.mean()

    def compute_spectral_loss(self, encoder, est_target, target):
        spect_est_target = encoder(est_target)
        spect_target = encoder(target)
        linear_loss = nn.functional.l1_loss(spect_est_target, spect_target)
        return linear_loss * self.loss_scale


class SingleSrcMultiScaleSpectral_L1_above_freq(SingleSrcMultiScaleSpectral):
    def __init__(
        self,
        n_filters=[2048, 1024, 512],
        windows_size=[2048, 1024, 512],
        hops_size=[512, 256, 128],
        alpha=None,
        loss_scale=1.0,
        sample_rate=24000,
        above_freq=1000,
        log_scale=False,
    ):
        super().__init__(n_filters, windows_size, hops_size, alpha)
        # Compute L1 loss between targets and sources on each real and imaginary values
        # We want to use l1 loss only above the specified frequency.
        # l1 loss will be calculated only on 'frequency' argument
        self.loss_scale = loss_scale
        self.over_freq_indices = []
        for each_window_size in windows_size:
            total_n_bins = int(1 + each_window_size / 2)
            stft_bins_freqs = (
                np.arange(0, total_n_bins) * sample_rate / each_window_size
            )  # make the array that contains each frequencis
            over_freq_index = np.where(stft_bins_freqs >= above_freq)[0][
                0
            ]  # to find where the starting frequency bin that exceeds 'above_freq'
            self.over_freq_indices.append(
                [over_freq_index, over_freq_index + total_n_bins]
            )
        self.log_scale = log_scale

    def forward(self, est_target, target):
        est_target = est_target.unsqueeze(1)
        target = target.unsqueeze(1)

        loss = torch.zeros(1, device=est_target.device)
        for idx, encoder in enumerate(self.encoders):
            loss += self.compute_spectral_loss(
                encoder, est_target, target, self.over_freq_indices[idx]
            )
        if self.log_scale:
            return torch.log(loss.mean() + 1e-6)
        else:
            return loss.mean()

    def compute_spectral_loss(self, encoder, est_target, target, over_freq_indices):
        spect_est_target = encoder(est_target)  # torch.Size([64, 1, 2, 1026, 278])
        freq_bin_size = spect_est_target.shape[-2]
        spect_target = encoder(target)
        linear_loss = nn.functional.l1_loss(
            spect_est_target[..., over_freq_indices[0] : freq_bin_size // 2, :],
            spect_target[..., over_freq_indices[0] : freq_bin_size // 2, :],
        )
        +nn.functional.l1_loss(
            spect_est_target[..., over_freq_indices[1] :, :],
            spect_target[..., over_freq_indices[1] :, :],
        )
        return linear_loss * self.loss_scale
