import torch
import numpy as np
import torchaudio
from asteroid.models.base_models import (
    BaseEncoderMaskerDecoder,
    _unsqueeze_to_3d,
    _shape_reconstructed,
)
from asteroid.utils.torch_utils import pad_x_to_y, jitable_shape
from asteroid_filterbanks.transforms import mag, magphase, from_magphase

from ..utils import my_magphase, normalize_mag_spec, denormalize_mag_spec


class BaseEncoderMaskerDecoder_output_maksed_tf(BaseEncoderMaskerDecoder):
    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__(encoder, masker, decoder, encoder_activation)

    def forward(self, wav):
        """Enc/Mask/Dec model forward
            with OUTPUT MASKED TF REPRESENTATION

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape), masked_tf_rep

class BaseEncoderMaskerDecoder_mixture_consistency(BaseEncoderMaskerDecoder):
    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__(encoder, masker, decoder, encoder_activation)

    def forward(self, wav):
        """Enc/Mask/Dec model forward with mixture consistent output

        References:
        [1] : Wisdom, Scott, et al. "Differentiable consistency constraints for improved deep speech enhancement." ICASSP 2019.
        [2] : Wisdom, Scott, et al. "Unsupervised sound separation using mixture invariant training." NeurIPS 2020.

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = _shape_reconstructed(pad_x_to_y(decoded, wav), shape)

        reconstructed = reconstructed + 1 / reconstructed.shape[1] * (
            wav - reconstructed.sum(dim=1, keepdim=True)
        )

        return reconstructed


class BaseEncoderMaskerDecoder_mixture_consistency_super_resolution(
    BaseEncoderMaskerDecoder
):
    def __init__(
        self,
        encoder,
        masker,
        decoder,
        sr_net,
        window_size=2048,
        above_freq=3000.0,
        sample_rate=24000,
        encoder_activation=None,
        db_normalize=False,
        sr_input_res=False,
        sr_out_mix_consistency=False,
    ):
        super().__init__(encoder, masker, decoder, encoder_activation)
        self.sr_net = sr_net
        self.over_freq_indices = []
        total_n_bins = int(1 + window_size / 2)
        stft_bins_freqs = (
            np.arange(0, total_n_bins) * sample_rate / window_size
        )  # make the array that contains each frequencis
        self.over_freq_index = np.where(stft_bins_freqs >= above_freq)[0][0]
        self.db_normalize = db_normalize
        self.sr_input_res = sr_input_res
        self.sr_out_mix_consistency = sr_out_mix_consistency

    def forward(self, wav):
        """Enc/Mask/Dec + Super-Resolution model forward

        References:
        [1] : Rixon, Joel, et al. "SFSRNet: Super-resolution for single-channel Audio Source Separation." AAAI 2022.

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        reconstructed = self.forward_pre(wav)

        sr_out_recon = self.forward_sr(wav, reconstructed)

        return sr_out_recon

    def forward_pre(self, wav):
        """Enc/Mask/Dec + Super-Resolution model forward

        References:
        [1] : Rixon, Joel, et al. "SFSRNet: Super-resolution for single-channel Audio Source Separation." AAAI 2022.

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)
        
        # forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = _shape_reconstructed(pad_x_to_y(decoded, wav), shape)

        reconstructed = reconstructed + 1 / reconstructed.shape[1] * (
            wav - reconstructed.sum(dim=1, keepdim=True)
        )

        return reconstructed

    def forward_sr(self, wav, reconstructed):
        """Enc/Mask/Dec + Super-Resolution model forward

        References:
        [1] : Rixon, Joel, et al. "SFSRNet: Super-resolution for single-channel Audio Source Separation." AAAI 2022.

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.
            reconstructed (torch.Tensor): output waveform tensor from self.forward_pre

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # forward
        tf_rep = self.forward_encoder(wav)

        out_est_stft = self.forward_encoder(reconstructed)

        tf_rep = tf_rep.unsqueeze(1)
        mix_mag = mag(tf_rep)
        if self.training:
            est_mag, est_phase = my_magphase(out_est_stft)
        else:
            est_mag, est_phase = magphase(out_est_stft)

        if self.db_normalize:
            mix_mag = normalize_mag_spec(
                torchaudio.functional.amplitude_to_DB(
                    mix_mag, multiplier=20.0, amin=1e-5, db_multiplier=1.0
                )
            )
            est_mag = normalize_mag_spec(
                torchaudio.functional.amplitude_to_DB(
                    est_mag, multiplier=20.0, amin=1e-5, db_multiplier=1.0
                )
            )

        heuristic_out = self.heuristic(mix_mag, est_mag)

        sr_out = self.sr_net(mix_mag, est_mag, heuristic_out)

        if self.sr_input_res:
            sr_out = sr_out + est_mag

        if self.db_normalize:
            sr_out = torchaudio.functional.DB_to_amplitude(
                denormalize_mag_spec(sr_out), ref=1.0, power=0.5
            )

        sr_out_stft = from_magphase(sr_out, est_phase)
        sr_out_decoded = self.forward_decoder(sr_out_stft)

        sr_out_recon = _shape_reconstructed(pad_x_to_y(sr_out_decoded, wav), shape)

        if self.sr_out_mix_consistency:
            sr_out_recon = sr_out_recon + 1 / sr_out_recon.shape[1] * (
                wav - sr_out_recon.sum(dim=1, keepdim=True)
            )

        return sr_out_recon

    def heuristic(self, mix_mag, est_mag):
        mix_sum_freq = mix_mag[..., : self.over_freq_index, :].sum(
            dim=-2, keepdim=True
        )  # which is described in SFSRNet heuristic
        est_sum_freq = est_mag[..., : self.over_freq_index, :].sum(dim=-2, keepdim=True)

        ratio = mix_sum_freq / (est_sum_freq + 1e-5)
        mix_high_freqs = mix_mag[..., self.over_freq_index :, :]

        heuristic_out = mix_high_freqs * ratio
        heuristic_out = torch.cat(
            [
                torch.zeros(
                    [
                        mix_high_freqs.shape[0],
                        heuristic_out.shape[1],
                        self.over_freq_index,
                        mix_high_freqs.shape[3],
                    ],
                    requires_grad=True,
                    device=heuristic_out.device,
                ),
                heuristic_out,
            ],
            dim=-2,
        )

        return heuristic_out
