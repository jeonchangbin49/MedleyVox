import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from asteroid.masknn import activations, norms
from asteroid.dsp.overlap_add import DualPathProcessing
from asteroid.utils import has_arg

from .pos_encoding import PositionalEncoding


class SepFormerTransformerLayer(nn.Module):
    """
    Transformer module used in [1].
    It is Multi-Head self-attention followed, activation and linear projection layer.
    Args:
        embed_dim (int): Number of input channels.
        n_heads (int): Number of attention heads.
        dim_ff (int): Number of neurons in the RNNs cell state.
            Defaults to 256. RNN here replaces standard FF linear layer in plain Transformer.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        activation (str, optional): activation function applied at the output of RNN.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        norm (str, optional): Type of normalization to use.
    References
        [1] Cem, Subakan et al. "Attention is all you need in Speech Separation."
        https://arxiv.org/abs/2010.13154, ICASSP (2021).
    """

    def __init__(
        self,
        embed_dim,
        n_heads,
        dim_ff=1024,
        dropout=0.0,
        activation="relu",
        # bidirectional=True,
        norm="gLN",
    ):
        super(SepFormerTransformerLayer, self).__init__()

        self.mha = MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        # self.mha = SelfAttention(embed_dim, heads=n_heads,causal=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # self.recurrent = nn.LSTM(embed_dim, dim_ff, bidirectional=bidirectional, batch_first=True)
        # ff_inner_dim = 2 * dim_ff if bidirectional else dim_ff
        ff_inner_dim = dim_ff
        self.activation = activations.get(activation)()
        self.norm_mha = norms.get(norm)(embed_dim)
        self.norm_ff = norms.get(norm)(embed_dim)

        # feedforward
        self.linear1 = nn.Linear(embed_dim, dim_ff)
        self.linear2 = nn.Linear(dim_ff, embed_dim)
        self.relu = nn.ReLU()

        self.pos_enc = PositionalEncoding(embed_dim, dropout=dropout)

        self.apply(self._init_weights)

    def forward(self, x):
        # tomha = x.permute(2, 0, 1) # x is batch, channels, seq_len

        # # mha is seq_len, batch, channels
        # # self-attention is applied
        # out = self.mha(tomha, tomha, tomha)[0] #[seq_len, batch, channels]
        # x = self.dropout(out.permute(1, 2, 0)) + x # [batch, channels, seq_len]
        # x = self.norm_mha(x)

        # # lstm is applied
        # out = self.linear(self.dropout(self.activation(self.recurrent(x.transpose(1, -1))[0])))
        # x = self.dropout(out.transpose(1, -1)) + x
        # return self.norm_ff(x)

        # x = [batch, channels, seq_len]    we use ola.intra_process or inter_process

        # batch, n_filters, chunk_size, n_chunks = x.size()
        # x = x.permute(0,3, 1, 2) # [batch, n_chunks, n_filters, chunk_size]
        # x = x.reshape([batch * n_chunks, n_filters, chunk_size])

        # using normal selfattention
        x = x.permute(2, 0, 1)  # [chunk_size, batch, n_filters]
        x_pos = self.pos_enc(x)  # [chunk_size, batch, n_filters]
        x_pos_norm = self.norm_mha(
            x_pos.permute(1, 2, 0)
        )  # [batch, n_filters, chunk_size]
        x_pos_norm = x_pos_norm.permute(2, 0, 1)  # [chunk_size, batch, n_filters]
        x_2_prime = self.mha(x_pos_norm, x_pos_norm, x_pos_norm)[
            0
        ]  # [chunk_size, batch, n_filters]
        x_2_prime = x_2_prime + x_pos  # [chunk_size, batch, n_filters]
        x_2_prime_norm = self.norm_ff(
            x_2_prime.permute(1, 2, 0)
        )  # [batch, n_filters, chunk_size]
        x_2_prime_norm = x_2_prime_norm.permute(
            2, 0, 1
        )  # [chunk_size, batch, n_filters]
        x_2_prime_norm_ff = self.linear2(
            self.relu(self.linear1(x_2_prime_norm))
        )  # [chunk_size, batch, n_filters]
        x_2_prime_norm_ff = (
            x_2_prime_norm_ff + x_2_prime
        )  # [chunk_size, batch, n_filters]
        output = x_2_prime_norm_ff + x  # [chunk_size, batch, n_filters]

        # using performer selfattention
        # x = x.permute(2, 0, 1) # [chunk_size, batch, n_filters]
        # x_pos = self.pos_enc(x) # [chunk_size, batch, n_filters]
        # x_pos_norm = self.norm_mha(x_pos.permute(1,2,0)) # [batch, n_filters, chunk_size]
        # # x_pos_norm = x_pos_norm.permute(2,0,1) # [chunk_size, batch, n_filters]
        # # x_2_prime = self.mha(x_pos_norm, x_pos_norm, x_pos_norm)[0] # [chunk_size, batch, n_filters]
        # x_2_prime = self.mha(x_pos_norm.transpose(1,2)) # [batch, chunk_size, n_filters]
        # x_2_prime = x_2_prime.permute(1, 0, 2) # [chunk_size, batch, n_filters]
        # x_2_prime = x_2_prime + x_pos # [chunk_size, batch, n_filters]
        # x_2_prime_norm = self.norm_ff(x_2_prime.permute(1,2,0)) # [batch, n_filters, chunk_size]
        # x_2_prime_norm = x_2_prime_norm.permute(2,0,1) # [chunk_size, batch, n_filters]
        # x_2_prime_norm_ff = self.linear2(self.relu(self.linear1(x_2_prime_norm))) # [chunk_size, batch, n_filters]
        # x_2_prime_norm_ff = x_2_prime_norm_ff + x_2_prime # [chunk_size, batch, n_filters]
        # output = x_2_prime_norm_ff + x # [chunk_size, batch, n_filters]

        return output.permute(1, 2, 0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # nn.init.kaiming_uniform_(module.weight,mode='fan_out')
            nn.init.uniform_(
                module.weight, a=-1 / module.in_features, b=1 / module.in_features
            )
            # module.bias.data.fill_(0.01)


class SepFormer(nn.Module):
    """SepFormer introduced in [1].
    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        n_heads (int): Number of attention heads.
        ff_hid (int): Number of neurons in the RNNs cell state.
            Defaults to 256.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use.
        ff_activation (str, optional): activation function applied at the output of RNN.
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        dropout (float, optional): Dropout ratio, must be in [0,1].
    References
        [1] Cem, Subakan et al. "Attention is all you need in Speech Separation."
        https://arxiv.org/abs/2010.13154, ICASSP (2021).
    """

    def __init__(
        self,
        in_chan,
        n_src,
        n_heads=8,
        ff_hid=1024,
        chunk_size=250,
        hop_size=None,
        n_repeats=8,
        n_blocks=2,
        norm_type="gLN",
        ff_activation="relu",
        mask_act="relu",
        bidirectional=True,
        dropout=0,
    ):
        super(SepFormer, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.n_heads = n_heads
        self.ff_hid = ff_hid
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_blocks = n_blocks
        self.n_src = n_src
        self.norm_type = norm_type
        self.ff_activation = ff_activation
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.dropout = dropout

        # self.mha_in_dim = math.ceil(self.in_chan / self.n_heads) * self.n_heads
        self.mha_in_dim = ff_hid
        # if self.in_chan % self.n_heads != 0:
        #     warnings.warn(
        #         f"SepFormer input dim ({self.in_chan}) is not a multiple of the number of "
        #         f"heads ({self.n_heads}). Adding extra linear layer at input to accomodate "
        #         f"(size [{self.in_chan} x {self.mha_in_dim}])"
        #     )
        #     self.input_layer = nn.Linear(self.in_chan, self.mha_in_dim)
        # else:
        #     self.input_layer = None

        self.in_norm = norms.get(norm_type)(self.in_chan)
        self.sepformer_input_layer = nn.Linear(self.in_chan, self.mha_in_dim)

        # self.in_norm = norms.get(norm_type)(self.mha_in_dim)
        self.ola = DualPathProcessing(self.chunk_size, self.hop_size)

        # # Succession of SepFormer blocks.
        # self.sepformer_layers = nn.ModuleList([])
        # self.intra_layers = nn.ModuleList([])
        # self.inter_layers = nn.ModuleList([])
        # for n_block in range(self.n_blocks):
        #     self.sepformer_layers.append
        #     self.intra_layers.append(
        #         nn.ModuleList(
        #             [SepFormerTransformerLayer(
        #                     self.mha_in_dim,
        #                     self.n_heads,
        #                     self.ff_hid,
        #                     self.dropout,
        #                     self.ff_activation,
        #                     self.norm_type,
        #                 ),
        #             ]
        #         ))
        self.sepformer_layers = nn.ModuleList([])
        for n_block in range(self.n_blocks):
            sepformer_block = nn.ModuleList([])
            intra_layers = nn.ModuleList([])
            inter_layers = nn.ModuleList([])
            for repeat in range(self.n_repeats):
                intra_layers.append(
                    SepFormerTransformerLayer(
                        self.mha_in_dim,
                        self.n_heads,
                        self.ff_hid,
                        self.dropout,
                        self.ff_activation,
                        self.norm_type,
                    )
                )
            for repeat in range(self.n_repeats):
                inter_layers.append(
                    SepFormerTransformerLayer(
                        self.mha_in_dim,
                        self.n_heads,
                        self.ff_hid,
                        self.dropout,
                        self.ff_activation,
                        self.norm_type,
                    )
                )
            sepformer_block.append(intra_layers)
            sepformer_block.append(inter_layers)
            self.sepformer_layers.append(sepformer_block)

        # sepformer_layers = [[sepformer_block], [sepformer_block]]
        # sepformer_block = [[SepFormerTransformerLayer * 8], [SepFormerTransformerLayer * 8]]
        net_out_conv = nn.Conv2d(self.mha_in_dim, n_src * self.in_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        # self.net_out = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Tanh())
        # self.net_gate = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Sigmoid())

        self.sepformer_last_layer1 = nn.Linear(self.in_chan, self.in_chan)
        self.relu = nn.ReLU()
        self.sepformer_last_layer2 = nn.Linear(self.in_chan, self.in_chan)

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

        self.apply(self._init_weights)

    def forward(self, mixture_w):
        r"""Forward.
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$
        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        # if self.input_layer is not None:
        #     mixture_w = self.input_layer(mixture_w.transpose(1, 2)).transpose(1, 2)
        mixture_w = self.in_norm(mixture_w)  # [batch, bn_chan, n_frames]
        n_orig_frames = mixture_w.shape[-1]

        mixture_w = self.sepformer_input_layer(mixture_w.transpose(1, -1)).transpose(
            1, -1
        )  # [batch, bn_chan, n_frames]

        mixture_w = self.ola.unfold(mixture_w)
        batch, n_filters, self.chunk_size, n_chunks = mixture_w.size()

        # for layer_idx in range(len(self.layers)):
        #     intra, inter = self.layers[layer_idx]
        #     mixture_w = self.ola.intra_process(mixture_w, intra)
        #     mixture_w = self.ola.inter_process(mixture_w, inter)
        for block_idx in range(len(self.sepformer_layers)):
            block = self.sepformer_layers[block_idx]
            intra_blocks = block[0]  # intra or inter layer?
            for transformer_idx in range(self.n_repeats):
                mixture_w = self.ola.intra_process(
                    mixture_w, intra_blocks[transformer_idx]
                )
            inter_blocks = block[1]
            for transformer_idx in range(self.n_repeats):
                mixture_w = self.ola.inter_process(
                    mixture_w, inter_blocks[transformer_idx]
                )

        output = self.first_out(mixture_w)
        output = output.reshape(
            batch * self.n_src, self.in_chan, self.chunk_size, n_chunks
        )
        output = self.ola.fold(output, output_size=n_orig_frames)

        # output = self.net_out(output) * self.net_gate(output) # no gating in SepFormer
        # Compute mask
        output = output.reshape(batch, self.n_src, self.in_chan, -1)

        output = self.sepformer_last_layer2(
            self.relu(self.sepformer_last_layer1(output.transpose(-1, -2)))
        ).transpose(-1, -2)

        est_mask = self.output_act(output)
        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "ff_hid": self.ff_hid,
            "n_heads": self.n_heads,
            "chunk_size": self.chunk_size,
            "hop_size": self.hop_size,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "ff_activation": self.ff_activation,
            "mask_act": self.mask_act,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
        }
        return config

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # nn.init.kaiming_uniform_(module.weight,mode='fan_out')
            nn.init.uniform_(
                module.weight, a=-1 / module.in_features, b=1 / module.in_features
            )
            # module.bias.data.fill_(0.01)
