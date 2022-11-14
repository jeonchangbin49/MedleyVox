# Implementation of Rixon, Joel, et al. "SFSRNet: Super-resolution for single-channel Audio Source Separation." AAAI 2022.
import torch
import torch.nn as nn
from asteroid.masknn import norms


class SFSRNet(nn.Module):
    def __init__(self, n_src=2, norm_type="gLN"):
        super().__init__()
        # convolution
        input_channels = 1 + n_src * 2
        conv_norm = norms.get(norm_type)

        self.conv_1 = nn.Conv2d(input_channels, 128, 5, padding=5 // 2)
        nn.init.kaiming_normal_(self.conv_1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv_1.bias, 0.01)
        self.ln_1 = conv_norm(128)
        self.conv_2 = nn.Conv2d(128, 256, 9, padding=9 // 2)
        nn.init.kaiming_normal_(self.conv_2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv_2.bias, 0.01)
        self.ln_2 = conv_norm(256)
        self.conv_3 = nn.Conv2d(256, 128, 11, padding=11 // 2)
        nn.init.kaiming_normal_(self.conv_3.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv_3.bias, 0.01)
        self.ln_3 = conv_norm(128)
        self.conv_4 = nn.Conv2d(128, n_src, 11, padding=11 // 2)
        nn.init.kaiming_normal_(self.conv_4.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv_4.bias, 0.01)
        self.ln_4 = conv_norm(n_src)

        self.relu = nn.ReLU()

    def forward(self, mix_mag, est_mag, heuristic_out):
        input = torch.cat([mix_mag, est_mag, heuristic_out], dim=1)
        # convolution
        out = self.ln_2(
            self.relu(self.conv_2(self.ln_1(self.relu(self.conv_1(input)))))
        )
        out = self.ln_4(self.relu(self.conv_4(self.ln_3(self.relu(self.conv_3(out))))))

        return out


class SFSRNet_ConvNext(nn.Module):
    def __init__(self, n_src=2, norm_type="gLN"):
        super().__init__()
        # convolution
        input_channels = 1 + n_src * 2
        conv_norm = norms.get(norm_type)

        self.conv_0 = nn.Conv2d(
            input_channels,
            96,
            5,
            padding=5 // 2,
        )
        nn.init.kaiming_normal_(self.conv_0.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv_0.bias, 0.01)

        # 1st ConvNext block
        self.conv_1 = nn.Conv2d(96, 96, 7, padding=7 // 2, groups=96)
        nn.init.kaiming_normal_(self.conv_1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv_1.bias, 0.01)

        self.ln_1 = conv_norm(96)

        self.conv_2 = nn.Conv2d(
            96,
            384,
            1,
        )
        nn.init.kaiming_normal_(self.conv_2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv_2.bias, 0.01)

        self.conv_3 = nn.Conv2d(
            384,
            96,
            1,
        )
        nn.init.kaiming_normal_(self.conv_3.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv_3.bias, 0.01)

        # 2nd ConvNext block
        self.conv_4 = nn.Conv2d(96, 96, 1, groups=96)
        nn.init.kaiming_normal_(self.conv_4.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv_4.bias, 0.01)

        self.ln_4 = conv_norm(96)

        self.conv_5 = nn.Conv2d(
            96,
            384,
            1,
        )
        nn.init.kaiming_normal_(self.conv_5.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv_5.bias, 0.01)

        self.conv_6 = nn.Conv2d(
            384,
            96,
            1,
        )
        nn.init.kaiming_normal_(self.conv_6.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv_6.bias, 0.01)

        # output block
        self.conv_out = nn.Conv2d(
            96,
            n_src,
            1,
        )
        nn.init.kaiming_normal_(
            self.conv_out.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.conv_out.bias, 0.01)
        self.ln_out = conv_norm(n_src)

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, mix_mag, est_mag, heuristic_out):
        input = torch.cat([mix_mag, est_mag, heuristic_out], dim=1)
        # convolution
        out = self.gelu(self.conv_0(input))
        out_1 = self.conv_3(self.gelu(self.conv_2(self.ln_1(self.conv_1(out)))))
        out_2 = self.conv_6(self.gelu(self.conv_5(self.ln_4(self.conv_4(out + out_1)))))
        # out_3 = self.relu(self.conv_out(out_1 + out_2))
        out_3 = self.relu(self.ln_out(self.conv_out(out_1 + out_2)))

        return out_3
