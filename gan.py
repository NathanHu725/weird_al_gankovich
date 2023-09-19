import numpy as np 
import torch
import torch.nn as nn


class Transpose1dLayer(nn.Module):    
    """
    Taken from wavegan paper
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=11,
        upsample=None,
        output_padding=1,
        use_batch_norm=False,
    ):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample
        reflection_pad = nn.ConstantPad1d(kernel_size // 2, value=0)
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        conv1d.weight.data.normal_(0.0, 0.02)
        Conv1dTrans = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        batch_norm = nn.BatchNorm1d(out_channels)
        if self.upsample:
            operation_list = [reflection_pad, conv1d]
        else:
            operation_list = [Conv1dTrans]

        if use_batch_norm:
            operation_list.append(batch_norm)
        self.transpose_ops = nn.Sequential(*operation_list)

    def forward(self, x):
        if self.upsample:
            # recommended by wavgan paper to use nearest upsampling
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode="nearest")
        return self.transpose_ops(x)


class weirdAlGenerator(nn.Module):

    def __init__(
        self, 
        max_token_len=512, 
        num_channels =1, 
        output_dim=1024
    ):
        super(weirdAlGenerator, self).__init__()

        self.main(
            nn.Linear(max_token_len, 4 * 4 * 1024),
            nn.BatchNorm1d(num_features=1024),
            Transpose1dLayer(1024, 512, 25, stride=1),
        )