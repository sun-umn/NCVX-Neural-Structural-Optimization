import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fun


class CNNModel_torch(nn.Module):
    def __init__(
        self,
        latent_size=128,
        dense_channels=32,
        conv_filters=(128, 64, 32, 16, 1),
        kernel_size=(5, 5),
        dense_init_scale=1.0,
    ):
        super().__init__()
        resizes = (1, 1, 2, 2, 1)

        if len(resizes) != len(conv_filters):
            raise ValueError('resizes and filters must be same size')

        total_resize = int(np.prod(resizes))
        self.h = h = 20 // total_resize
        self.w = w = 60 // total_resize
        self.dense_channels = dense_channels
        self.resizes = resizes
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size

        inputs = torch.randn(1, latent_size)
        filters = dense_channels * h * w
        nn.init.orthogonal_(
            inputs, gain=dense_init_scale * np.sqrt(max(filters / latent_size, 1))
        )

        # Optimization variables used in PyGRANSO
        self.dense = nn.Linear(latent_size, filters)
        self.conv = nn.ModuleList()
        self.conv_bn = nn.ModuleList()
        # not used in the CNN model. It's a dummy variable
        # used in PyGRANSO that has to be defined there,
        # as PyGRANSO will read all parameters from the nn.parameters()
        self.U = torch.nn.Parameter(torch.randn(60 * 20, 1))

        for in_channels, out_channels in zip(
            (dense_channels, 128, 64, 32, 16), conv_filters
        ):
            self.conv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    padding='same',
                )
            )
            self.conv_bn.append(nn.BatchNorm2d(num_features=in_channels))

    def forward(self, x):
        x = self.dense(x)
        x = x.reshape((1, self.dense_channels, self.h, self.w))

        for resize, filters, i in zip(
            self.resizes, self.conv_filters, list(range(len(self.conv_filters)))
        ):
            x = Fun.tanh(x)
            x = Fun.upsample(x, scale_factor=resize, mode='bilinear')
            x = self.conv_bn[i](x)
            x = self.conv[i](x)

        return x
