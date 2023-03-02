import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as Fun


def set_seed(manualSeed):  # noqa
    """
    Function to set the seed
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)


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
        self.scale = scale
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
        conv_filters=(128, 64, 32, 16, 1),
        offset_scale=10.0,
        kernel_size=(5, 5),
        latent_scale=1.0,
        dense_init_scale=1.0,
        train_u_matrix=False,
    ):
        super().__init__()

        # Raise an error if the resizes are not equal to the convolutional
        # filters
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
        self.kernel_size = kernel_size
        self.latent_size = latent_size
        self.dense_init_scale = dense_init_scale
        self.offset_scale = offset_scale
        self.conv_filters = conv_filters

        # Create the filters
        filters = dense_channels * self.h * self.w

        # Create the u_matrix vector
        if train_u_matrix:
            distribution = torch.distributions.uniform.Uniform(-5500.0, 500.0)
            sample = distribution.sample(torch.Size([len(args["freedofs"])]))
            self.u_matrix = nn.Parameter(sample.double())

        # Create the first dense layer
        self.dense = nn.Linear(latent_size, filters)

        # Create the gain for the initializer
        gain = self.dense_init_scale * np.sqrt(max(filters / latent_size, 1.0))
        nn.init.orthogonal_(self.dense.weight, gain=gain)

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
            torch.nn.init.kaiming_normal_(
                convolution_layer.weight, mode="fan_in", nonlinearity="leaky_relu"
            )

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
        self.z = torch.normal(mean=0.0, std=1.0, size=(1, latent_size))
        self.z = nn.Parameter(self.z)

        # STE function
        # self.ste = STEFunction

    def forward(self, x=None):  # noqa

        # Create the model
        output = self.dense(self.z)
        output = output.reshape((1, self.dense_channels, self.h, self.w))

        layer_loop = zip(self.resizes, self.conv_filters)
        for idx, (resize, filters) in enumerate(layer_loop):
            # output = torch.tanh(output)
            output = nn.LeakyReLU()(output)
            # output = nn.ReLU()(output)
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

        # Squeeze the result in the first axis just like in the
        # tensorflow code
        output = torch.squeeze(output)
        # output = self.ste.apply(output)

        return output


class MultiMaterialModel(nn.Module):
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
        conv_filters=(128, 64, 32, 16),
        offset_scale=10.0,
        kernel_size=(5, 5),
        latent_scale=1.0,
        dense_init_scale=1.0,
    ):
        super().__init__()

        # Raise an error if the resizes are not equal to the convolutional
        # filters
        self.num_materials = args["num_materials"]
        self.conv_filters = conv_filters + (self.num_materials + 1,)
        if len(resizes) != len(self.conv_filters):
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
        self.kernel_size = kernel_size
        self.latent_size = latent_size
        self.dense_init_scale = dense_init_scale
        self.offset_scale = offset_scale

        # Create the filters
        filters = dense_channels * self.h * self.w

        # Create the first dense layer
        self.dense = nn.Linear(latent_size, filters)

        # Create the gain for the initializer
        gain = self.dense_init_scale * np.sqrt(max(filters / latent_size, 1.0))
        nn.init.orthogonal_(self.dense.weight, gain=gain)

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
        offset_filters_tuple = self.conv_filters[:-1]
        offset_filters = dense_channels_tuple + offset_filters_tuple

        for resize, in_channels, out_channels in zip(
            self.resizes, offset_filters, self.conv_filters
        ):
            convolution_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                padding="same",
            )
            torch.nn.init.xavier_normal_(
                convolution_layer.weight, gain=nn.init.calculate_gain("relu")
            )

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
        self.z = torch.normal(mean=0.0, std=1.0, size=(1, latent_size))
        self.z = nn.Parameter(self.z)

        # STE function
        # self.ste = STEFunction

    def forward(self, x=None):  # noqa

        # Create the model
        output = self.dense(self.z)
        output = output.reshape((1, self.dense_channels, self.h, self.w))

        layer_loop = zip(self.resizes, self.conv_filters)
        for idx, (resize, filters) in enumerate(layer_loop):
            output = nn.ReLU6()(output)

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

        # Squeeze the result in the first axis just like in the
        # tensorflow code
        output = torch.squeeze(output)

        return output


class TopNetPyGranso(nn.Module):
    def __init__(
        self,
        numLayers,
        numNeuronsPerLyr,
        nelx,
        nely,
        numMaterials,
        symXAxis,
        symYAxis,
        seed,
    ):
        self.inputDim = 2
        # x and y coordn of the point
        self.outputDim = numMaterials + 1
        # if material A/B/.../void at the point
        self.nelx = nelx
        self.nely = nely
        self.symXAxis = symXAxis
        self.symYAxis = symYAxis
        self.numLayers = numLayers
        self.numNeuronsPerLyr = numNeuronsPerLyr
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = self.inputDim
        set_seed(seed)

        for lyr in range(numLayers):
            l = nn.Linear(current_dim, numNeuronsPerLyr)
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
            self.layers.append(l)
            current_dim = numNeuronsPerLyr

        self.layers.append(nn.Linear(current_dim, self.outputDim))
        self.bnLayer = nn.ModuleList()
        for lyr in range(numLayers):
            self.bnLayer.append(nn.BatchNorm1d(numNeuronsPerLyr))

        xy, self.nonDesignIdx = self.generatePoints(nelx, nely, 1, None)
        self.xy = nn.Parameter(xy)

    def generatePoints(
        self, nx, ny, resolution=1, nonDesignRegion=None
    ):  # generate points in elements
        ctr = 0
        xy = np.zeros((resolution * nx * resolution * ny, 2))
        nonDesignIdx = []
        for i in range(resolution * nx):
            for j in range(resolution * ny):
                xy[ctr, 0] = (i + 0.5) / resolution
                xy[ctr, 1] = (j + 0.5) / resolution
                if nonDesignRegion is not None:
                    if (
                        (xy[ctr, 0] < nonDesignRegion["x<"])
                        and (xy[ctr, 0] > nonDesignRegion["x>"])
                        and (xy[ctr, 1] < nonDesignRegion["y<"])
                        and (xy[ctr, 1] > nonDesignRegion["y>"])
                    ):
                        nonDesignIdx.append(ctr)
                ctr += 1
        xy = torch.tensor(xy).view(-1, 2)
        return xy, nonDesignIdx

    def forward(self, x=None, fixedIdx=None):
        x = self.xy
        m = nn.ReLU6()
        #
        ctr = 0
        if self.symYAxis:
            xv = 0.5 * self.nelx + torch.abs(x[:, 0] - 0.5 * self.nelx)
        else:
            xv = x[:, 0]
        if self.symXAxis:
            yv = 0.5 * self.nely + torch.abs(x[:, 1] - 0.5 * self.nely)
        else:
            yv = x[:, 1]

        x = torch.transpose(torch.stack((xv, yv)), 0, 1)
        for layer in self.layers[:-1]:
            x = m(self.bnLayer[ctr](layer(x)))
            ctr += 1
        out = 1e-4 + torch.softmax(self.layers[-1](x), dim=1)

        if fixedIdx is not None:
            out[fixedIdx, 0] = 0.95
            out[fixedIdx, 1:] = 0.01
            # fixed Idx removes region

        # Reshaping the last layer to have the same outputs as the
        # multi-material paper
        out = (
            out.reshape(self.nely, self.nelx, self.outputDim)
            .transpose(0, 2)
            .transpose(1, 2)
        )
        return out


class UMatrixModel(nn.Module):
    """
    Class that will simply implement a u matrix for us to
    train
    """

    def __init__(self, args, uniform_lower_bound, uniform_upper_bound):  # noqa
        super().__init__()
        self.uniform_upper_bound = uniform_upper_bound
        self.uniform_lower_bound = uniform_lower_bound

        # Initialize U from a uniform distribution
        distribution = torch.distributions.uniform.Uniform(
            self.uniform_lower_bound, self.uniform_upper_bound
        )
        sample = distribution.sample(torch.Size([len(args["freedofs"])]))
        self.u_matrix = nn.Parameter(sample.double())

    def forward(self, x=None):  # noqa
        return self.u_matrix


##### Multi-Material MM Neural Network Code #####
class TopNet(nn.Module):
    def __init__(
        self, numLayers, numNeuronsPerLyr, nelx, nely, numMaterials, symXAxis, symYAxis
    ):
        self.inputDim = 2
        # x and y coordn of the point
        self.outputDim = numMaterials + 1
        # if material A/B/.../void at the point
        self.nelx = nelx
        self.nely = nely
        self.symXAxis = symXAxis
        self.symYAxis = symYAxis
        self.numLayers = numLayers
        self.numNeuronsPerLyr = numNeuronsPerLyr
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = self.inputDim
        set_seed(1)
        # NN are seeded manually

        for lyr in range(numLayers):
            l = nn.Linear(current_dim, numNeuronsPerLyr)
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
            self.layers.append(l)
            current_dim = numNeuronsPerLyr

        self.layers.append(nn.Linear(current_dim, self.outputDim))
        self.bnLayer = nn.ModuleList()
        for lyr in range(numLayers):
            self.bnLayer.append(nn.BatchNorm1d(numNeuronsPerLyr))

    def forward(self, x, fixedIdx=None):
        # activations ReLU, ReLU6, ELU, SELU, PReLU, LeakyReLU,
        # Sigmoid, Tanh, LogSigmoid, Softplus, Softsign, TanhShrink,
        # Softmin, Softmax
        m = nn.ReLU6()
        #
        ctr = 0
        if self.symYAxis:
            xv = 0.5 * self.nelx + torch.abs(x[:, 0] - 0.5 * self.nelx)
        else:
            xv = x[:, 0]
        if self.symXAxis:
            yv = 0.5 * self.nely + torch.abs(x[:, 1] - 0.5 * self.nely)
        else:
            yv = x[:, 1]

        x = torch.transpose(torch.stack((xv, yv)), 0, 1)
        for layer in self.layers[:-1]:
            x = m(self.bnLayer[ctr](layer(x)))
            ctr += 1
        out = 1e-4 + torch.softmax(self.layers[-1](x), dim=1)

        if fixedIdx is not None:
            out[fixedIdx, 0] = 0.95
            out[fixedIdx, 1:] = 0.01
            # fixed Idx removes region
        return out

    def getWeights(self):  # stats about the NN
        modelWeights = []
        modelBiases = []
        for lyr in self.layers:
            modelWeights.extend(lyr.weight.data.cpu().view(-1).numpy())
            modelBiases.extend(lyr.bias.data.cpu().view(-1).numpy())
        return modelWeights, modelBiases


class MultiMaterialCNN(nn.Module):
    """
    Class that implements the CNN model from the structural
    optimization paper
    """

    def __init__(  # noqa
        self,
        nelx,
        nely,
        num_materials,
        latent_size=128,
        dense_channels=32,
        resizes=(1, 2, 2, 2, 1),
        conv_filters=(128, 64, 32, 16),
        offset_scale=10.0,
        kernel_size=(5, 5),
        latent_scale=1.0,
        dense_init_scale=1.0,
    ):
        super().__init__()
        set_seed(0)

        # Raise an error if the resizes are not equal to the convolutional
        # filters
        self.num_materials = num_materials
        self.nelx = nelx
        self.nely = nely
        self.conv_filters = conv_filters + (self.num_materials + 1,)
        if len(resizes) != len(self.conv_filters):
            raise ValueError("resizes and filters are not the same!")

        total_resize = int(np.prod(resizes))
        self.h = int(torch.div(nely, total_resize, rounding_mode="floor").item())
        self.w = int(torch.div(nelx, total_resize, rounding_mode="floor").item())
        self.dense_channels = dense_channels
        self.resizes = resizes
        self.kernel_size = kernel_size
        self.latent_size = latent_size
        self.dense_init_scale = dense_init_scale
        self.offset_scale = offset_scale

        # Create the filters
        filters = dense_channels * self.h * self.w

        # Create the first dense layer
        self.dense = nn.Linear(latent_size, filters)

        # Create the gain for the initializer
        gain = self.dense_init_scale * np.sqrt(max(filters / latent_size, 1.0))
        nn.init.orthogonal_(self.dense.weight, gain=gain)

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
        offset_filters_tuple = self.conv_filters[:-1]
        offset_filters = dense_channels_tuple + offset_filters_tuple

        for resize, in_channels, out_channels in zip(
            self.resizes, offset_filters, self.conv_filters
        ):
            convolution_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                padding="same",
            )
            torch.nn.init.kaiming_normal_(
                convolution_layer.weight, mode="fan_in", nonlinearity="leaky_relu"
            )

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
        self.z = torch.normal(mean=0.0, std=1.0, size=(1, latent_size))
        self.z = nn.Parameter(self.z)

        # STE function
        # self.ste = STEFunction

    def forward(self, x=None):  # noqa

        # Create the model
        output = self.dense(self.z)
        output = output.reshape((1, self.dense_channels, self.h, self.w))

        layer_loop = zip(self.resizes, self.conv_filters)
        for idx, (resize, filters) in enumerate(layer_loop):
            output = nn.LeakyReLU()(output)

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

        # Squeeze the result in the first axis just like in the
        # tensorflow code
        output = torch.squeeze(output)

        # Add the soft max layer
        softmax = nn.Softmax()
        output = softmax(output) + 1e-4
        output = output.transpose(0, 2)
        output = output.reshape(self.nelx * self.nely, self.num_materials + 1)

        return output
