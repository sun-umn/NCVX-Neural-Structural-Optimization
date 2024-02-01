import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fun


def set_seed(seed):
    """
    Function to set the seed for the run
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_seeded_random_variable(latent_size, seed):
    """
    Want to try an experiment that could help the network
    generalize a bit better. Right now the best designs
    depend on the random seed that is being initialized
    """
    set_seed(seed)
    return torch.normal(mean=0.0, std=1.0, size=(3, latent_size))


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return Fun.hardtanh(grad_output)


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
        var, mean = torch.var_mean(x, unbiased=False)
        net = x
        net = net - mean
        net = net * torch.rsqrt(var + self.epsilon)
        return net


# Create a layer to add offsets
class AddOffset(nn.Module):
    """
    Class that adds the weights / bias offsets & is
    trainable for the structural optimization code
    """

    def __init__(self, conv_channels, height, width, scale=10):  # noqa
        super().__init__()
        self.scale = torch.tensor(scale, requires_grad=True)
        self.conv_channels = conv_channels
        self.height = height
        self.width = width
        self.bias = nn.Parameter(
            torch.zeros(1, self.conv_channels, self.height, self.width),
            requires_grad=True,
        )

    def forward(self, x):  # noqa
        return x + (self.scale * self.bias)


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
        conv_filters=(128 * 2 // 1, 64 * 2 // 1, 32 * 2 // 1, 16 * 2 // 1, 8 * 2 // 1),
        offset_scale=10.0,
        kernel_size=(5, 5),
        latent_scale=1.0,
        dense_init_scale=1.0,
        random_seed=0,
        train_u_matrix=False,
    ):
        super().__init__()
        set_seed(random_seed)

        # Raise an error if the resizes are not equal to the convolutional
        # filteres
        if len(resizes) != len(conv_filters):
            raise ValueError("resizes and filters are not the same!")

        total_resize = int(np.prod(resizes))
        self.h = int(
            torch.div(args["nely"], total_resize, rounding_mode="floor").item()
        )
        self.w = int(
            torch.div(args["nelx"], total_resize, rounding_mode="floor").item()
        )
        self.dense_channels = dense_channels
        self.resizes = resizes
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.latent_size = latent_size
        self.dense_init_scale = dense_init_scale
        self.offset_scale = offset_scale

        # Create the filters
        filters = dense_channels * self.h * self.w

        # Create the u_matrix vector
        if train_u_matrix:
            distribution = torch.distributions.uniform.Uniform(-5500.0, 500.0)
            sample = distribution.sample(torch.Size([len(args["freedofs"])]))
            self.u_matrix = nn.Parameter(sample.double())

        # Create the first dense layer
        self.dense = nn.Linear(latent_size, filters)

        # # Create the gain for the initializer
        gain = self.dense_init_scale * np.sqrt(max(filters / latent_size, 1.0))
        nn.init.orthogonal_(self.dense.weight, gain=gain)
        # torch.nn.init.xavier_normal_(self.dense.weight, gain=gain)

        # Create the convoluational layers that will be used
        self.conv = nn.ModuleList()

        # Global normalization layers
        self.global_normalization = nn.ModuleList()

        # Trainable bias layer
        self.add_offset = nn.ModuleList()

        # Add the convolutional layers to the module list
        height = self.h
        width = self.w

        dense_channels_tuple = (dense_channels,)
        offset_filters_tuple = conv_filters[:-1]
        offset_filters = dense_channels_tuple + offset_filters_tuple

        for resize, in_channels, out_channels in zip(
            self.resizes, offset_filters, conv_filters
        ):
            convolution_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                padding="same",
            )
            # This was the best initialization
            torch.nn.init.kaiming_normal_(
                convolution_layer.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
            # torch.nn.init.xavier_normal_(convolution_layer.weight, gain=1.0)
            # torch.nn.init.xavier_uniform_(convolution_layer.weight, gain=1.2)
            self.conv.append(convolution_layer)
            self.global_normalization.append(GlobalNormalization())

            # TODO: Fix the offset layer
            height = height * resize
            width = width * resize
            offset_layer = AddOffset(
                scale=self.offset_scale,
                conv_channels=out_channels,
                height=height,
                width=width,
            )
            self.add_offset.append(offset_layer)

        # Set up z here otherwise it is not part of the leaf tensors
        self.z = get_seeded_random_variable(latent_size, random_seed)
        self.z = torch.mean(self.z, axis=0)

        self.z = nn.Parameter(self.z)

    def forward(self, x=None):  # noqa
        # Create the model
        output = self.dense(self.z)
        output = output.reshape((1, self.dense_channels, self.h, self.w))

        layer_loop = zip(self.resizes, self.conv_filters)
        for idx, (resize, filters) in enumerate(layer_loop):
            # output = torch.tanh(output)

            output = torch.sin(output)
            # output = nn.LeakyReLU()(output)
            # After a lot of investigation the outputs of the upsample need
            # to be reconfigured to match the same expectation as tensorflow
            # so we will do that here. Also, interpolate is teh correct
            # function to use here
            output = Fun.interpolate(
                output,
                scale_factor=resize,
                mode="bilinear",
                align_corners=False,
            )

            # Apply the normalization
            output = self.global_normalization[idx](output)

            # Apply the 2D convolution
            output = self.conv[idx](output)

            if self.offset_scale != 0:
                output = self.add_offset[idx](output)

        # # Squeeze the result in the first axis just like in the
        # # tensorflow code
        output = torch.mean(output, axis=1)  # Along the feature dimension
        output = torch.squeeze(output)

        return output
