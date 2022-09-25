import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fun


# Create a custom global normalization layer for pytorch
class GlobalNormalization(nn.Module):
    """
    Class that computes the global normalization that we
    saw in the structural optimization code
    """

    def __init__(self, epsilon=1e-6):  # noqa
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):  # noqa
        x = x - x.mean()
        x = x * torch.rsqrt(x.var() + self.epsilon)
        return x


# Create a layer to add offsets
class AddOffset(nn.Module):
    """
    Class that adds the weights / bias offsets & is
    trainable for the structural optimization code
    """

    def __init__(self, scale=1):  # noqa
        super().__init__()
        self.scale = scale

    def forward(self, x):  # noqa
        torch_scale = torch.tensor(self.scale)
        bias = nn.Parameter(torch.zeros_like(x) * torch_scale)
        return x + bias


class CNNModel(nn.Module):
    """
    Class that implements the CNN model from the structural
    optimization paper
    """

    def __init__(  # noqa
        self,
        args,
        latent_size=128,
        dense_channels=32,
        resizes=(1, 2, 2, 2, 1),
        conv_filters=(128, 64, 32, 16, 1),
        offset_scale=10,
        kernel_size=(5, 5),
        latent_scale=1.0,
        dense_init_scale=1.0,
    ):
        super().__init__()

        # Raise an error if the resizes are not equal to the convolutional
        # filteres
        if len(resizes) != len(conv_filters):
            raise ValueError("resizes and filters are not the same!")

        total_resize = int(np.prod(resizes))
        self.h = args["nely"] // total_resize
        self.w = args["nelx"] // total_resize
        self.dense_channels = dense_channels
        self.resizes = resizes
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.latent_size = latent_size
        self.dense_init_scale = dense_init_scale
        self.offset_scale = offset_scale

        # Create the filters
        filters = dense_channels * self.h * self.w

        # # TODO: After the run with the new U we will bring this back
        # # Set up the u vector that we will be minimizing although
        # # it is not a part of the model

        # Create the first dense layer
        self.dense = nn.Linear(latent_size, filters)

        # Create the gain for the initializer
        gain = self.dense_init_scale * np.sqrt(max(filters / latent_size, 1))
        nn.init.orthogonal_(self.dense.weight, gain=gain)

        # # Create a global normalization layer
        # self.global_normalization = GlobalNormalization()

        # # Offsetting layer as seen in their code
        # self.add_offset = AddOffset(scale=self.offset_scale)

        # Create the convoluational layers that will be used
        self.conv = nn.ModuleList()

        # Global normalization layers
        self.global_normalization = nn.ModuleList()

        # Trainable bias layer
        self.add_offset = nn.ParameterList()

        # Add the convolutional layers to the module list
        offset_filters = (dense_channels, 128, 64, 32, 16)
        for in_channels, out_channels in zip(offset_filters, conv_filters):
            convolution_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                padding="same",
            )
            torch.nn.init.xavier_uniform_(convolution_layer.weight)
            self.conv.append(convolution_layer)
            self.global_normalization.append(GlobalNormalization())

            # TODO: Fix the offset layer
            offset_layer = AddOffset(self.offset_scale)
            self.add_offset.append(offset_layer)

        # Set up x here otherwise it is not part of the leaf tensors
        self.z = torch.nn.Parameter(
            torch.normal(mean=torch.zeros((1, 128)), std=torch.ones((1, 128)))
        )

    def forward(self, x=None):  # noqa

        # Create the model
        output = self.dense(self.z)
        output = output.reshape((1, self.dense_channels, self.h, self.w))

        layer_loop = zip(self.resizes, self.conv_filters)
        for idx, (resize, filters) in enumerate(layer_loop):
            output = torch.tanh(output)

            # After a lot of investigation the outputs of the upsample need
            # to be reconfigured to match the same expectation as tensorflow
            # so we will do that here. Also, interpolate is teh correct
            # function to use here
            output = Fun.interpolate(output, scale_factor=resize, mode="bilinear")

            # Apply the normalization
            output = self.global_normalization[idx](output)

            # Apply the 2D convolution
            output = self.conv[idx](output)

            if self.offset_scale != 0:
                output = self.add_offset[idx](output)

        # Squeeze the result in the last axis just like in the
        # tensorflow code
        output = torch.squeeze(output)

        # I thought this was one of the outputs that we wanted to find
        # the gradient for so we can try with and with out this
        output.retain_grad()

        return output
