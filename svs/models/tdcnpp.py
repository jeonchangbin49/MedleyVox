"""

implementation of TDCN++ in [1], slightly modified version of asteroid implementation. 

 [1] : Kavalerov, Ilya et al. “Universal Sound Separation.” in WASPAA 2019

note::
    The differences wrt to ConvTasnet's TCN are:
    1. Channel wise layer norm instead of global
    2. Longer-range skip-residual connections from earlier repeat inputs
        to later repeat inputs after passing them through dense layer.
    3. Learnable scaling parameter after each dense layer. The scaling
        parameter for the second dense  layer  in  each  convolutional
        block (which  is  applied  rightbefore the residual connection) is
        initialized to an exponentially decaying scalar equal to 0.9**L,
        where L is the layer or block index.
"""

import torch
import torch.nn as nn
from asteroid.masknn.convolutional import TDConvNetpp


class TDConvNetpp_modified(TDConvNetpp):
    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="fgLN",
        mask_act="relu",
    ):
        super().__init__(
            in_chan,
            n_src,
            out_chan,
            n_blocks,
            n_repeats,
            bn_chan,
            hid_chan,
            skip_chan,
            conv_kernel_size,
            norm_type,
            mask_act,
        )

        del self.consistency

    def forward(self, mixture_w):
        r"""Forward.
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$
        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        output_copy = output

        skip_connection = 0.0
        for r in range(self.n_repeats):
            # Long range skip connection TDCNpp
            if r != 0:
                # Transform the input to repeat r-1 and add to new repeat inp
                output = self.dense_skip[r - 1](output_copy) + output

                # Copy this for later.
                """
                output.clone() instead output
                """
                output_copy = output.clone()

            for x in range(self.n_blocks):
                # Common to w. skip and w.o skip architectures
                i = r * self.n_blocks + x
                tcn_out = self.TCN[i](output)
                if self.skip_chan:
                    residual, skip = tcn_out
                    skip_connection = skip_connection + skip
                else:
                    residual, _ = tcn_out
                # Initialized exp decay scale factor TDCNpp for residual connections
                scale = self.scaling_param[r, x - 1] if x > 0 else 1.0
                residual = residual * scale
                output = output + residual
        # Use residual output when no skip connection
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)

        # weights = self.consistency(mask_inp.mean(-1))
        # weights = torch.nn.functional.softmax(weights, -1)

        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "out_chan": self.out_chan,
            "bn_chan": self.bn_chan,
            "hid_chan": self.hid_chan,
            "skip_chan": self.skip_chan,
            "conv_kernel_size": self.conv_kernel_size,
            "n_blocks": self.n_blocks,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "mask_act": self.mask_act,
        }
        return
