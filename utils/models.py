# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import autograd
import autograd.core
import autograd.numpy as np
from utils import topo_api
import tensorflow as tf

import torch.nn as nn
import torch.nn.functional as F
import torch

# requires tensorflow 2.0

layers = tf.keras.layers


def batched_topo_loss(params, envs):
  losses = [env.objective(params[i], volume_contraint=True)
            for i, env in enumerate(envs)]
  return np.stack(losses)


def convert_autograd_to_tensorflow(func):
  @tf.custom_gradient
  def wrapper(x):
    vjp, ans = autograd.core.make_vjp(func, x.numpy())
    return ans, vjp
  return wrapper


def set_random_seed(seed):
  if seed is not None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


class Model(tf.keras.Model):

  def __init__(self, seed=None, args=None):
    super().__init__()
    set_random_seed(seed)
    self.seed = seed
    self.env = topo_api.Environment(args)

  def loss(self, logits):
    # for our neural network, we use float32, but we use float64 for the physics
    # to avoid any chance of overflow.
    # add 0.0 to work-around bug in grad of tf.cast on NumPy arrays
    logits = 0.0 + tf.cast(logits, tf.float64)
    f = lambda x: batched_topo_loss(x, [self.env])
    losses = convert_autograd_to_tensorflow(f)(logits)
    return tf.reduce_mean(losses)


class PixelModel(Model):

  def __init__(self, seed=None, args=None):
    super().__init__(seed, args)
    shape = (1, self.env.args['nely'], self.env.args['nelx'])
    z_init = np.broadcast_to(args['volfrac'] * args['mask'], shape)
    self.z = tf.Variable(z_init, trainable=True)

  def call(self, inputs=None):
    return self.z


def global_normalization(inputs, epsilon=1e-6):
  mean, variance = tf.nn.moments(inputs, axes=list(range(len(inputs.shape))))
  net = inputs
  net -= mean
  net *= tf.math.rsqrt(variance + epsilon)
  return net

def normalization(inputs, epsilon=1e-6):
  variance, mean = torch.var_mean(inputs)
  x = inputs
  x -= mean
  x *= torch.rsqrt(variance + epsilon)
  return x

def UpSampling2D(factor):
  return layers.UpSampling2D((factor, factor), interpolation='bilinear')

def UpSampling(scale_factor):
  return nn.Upsample(scale_factor = scale_factor, mode='bilinear')

def Conv2D(filters, kernel_size, **kwargs):
  return layers.Conv2D(filters, kernel_size, padding='same', **kwargs)


class AddOffset(layers.Layer):

  def __init__(self, scale=1):
    super().__init__()
    self.scale = scale

  def build(self, input_shape):
    self.bias = self.add_weight(
        shape=input_shape, initializer='zeros', trainable=True, name='bias')

  def call(self, inputs):
    return inputs + self.scale * self.bias


class CNNModel(Model):

  def __init__(
      self,
      seed=0,
      args=None,
      latent_size=128,
      dense_channels=32,
      resizes=(1, 2, 2, 2, 1),
      conv_filters=(128, 64, 32, 16, 1),
      offset_scale=10,
      kernel_size=(5, 5),
      latent_scale=1.0,
      dense_init_scale=1.0,
      activation=tf.nn.tanh,
      conv_initializer=tf.initializers.VarianceScaling,
      normalization=global_normalization,
  ):
    super().__init__(seed, args)

    if len(resizes) != len(conv_filters):
      raise ValueError('resizes and filters must be same size')

    activation = layers.Activation(activation)

    total_resize = int(np.prod(resizes))
    h = self.env.args['nely'] // total_resize
    w = self.env.args['nelx'] // total_resize

    net = inputs = layers.Input((latent_size,), batch_size=1)
    filters = h * w * dense_channels
    dense_initializer = tf.initializers.orthogonal(
        dense_init_scale * np.sqrt(max(filters / latent_size, 1)))
    net = layers.Dense(filters, kernel_initializer=dense_initializer)(net)
    net = layers.Reshape([h, w, dense_channels])(net)

    for resize, filters in zip(resizes, conv_filters):
      net = activation(net)
      net = UpSampling2D(resize)(net)
      net = normalization(net)
      net = Conv2D(
          filters, kernel_size, kernel_initializer=conv_initializer)(net)
      if offset_scale != 0:
        net = AddOffset(offset_scale)(net)

    outputs = tf.squeeze(net, axis=[-1])

    self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
    self.z = self.add_weight(
        shape=inputs.shape, initializer=latent_initializer, name='z')

  def call(self, inputs=None):
    return self.core_model(self.z)

  
class CNNModel_torch(nn.Module):

  def __init__(
      self,
      seed=0,
      args=None,
      latent_size=128,
      dense_channels=32,
      resizes=(1, 2, 2, 2, 1),
      conv_filters=(128, 64, 32, 16, 1),
      offset_scale=10,
      kernel_size=(5, 5),
      latent_scale=1.0,
      dense_init_scale=1.0,
      activation=nn.Tanh,
      conv_initializer=tf.initializers.VarianceScaling,
      normalization=global_normalization,
  ):
    # super().__init__(seed, args)
    super().__init__()
    self.env = topo_api.Environment(args)

    if len(resizes) != len(conv_filters):
      raise ValueError('resizes and filters must be same size')

    # self.activation = nn.Tanh

    total_resize = int(np.prod(resizes))
    self.h = h = self.env.args['nely'] // total_resize
    self.w = w = self.env.args['nelx'] // total_resize
    self.dense_channels = dense_channels
    self.resizes = resizes
    self.conv_filters = conv_filters
    self.kernel_size = kernel_size

    # net = inputs = layers.Input((latent_size,), batch_size=1)
    inputs = torch.randn(1,latent_size)

    filters = dense_channels * h * w 

    nn.init.orthogonal_(inputs,gain=dense_init_scale * np.sqrt(max(filters / latent_size, 1)))

    # dense_initializer = tf.initializers.orthogonal(
    #     dense_init_scale * np.sqrt(max(filters / latent_size, 1)))
    # net = layers.Dense(filters, kernel_initializer=dense_initializer)(net)
    self.dense = nn.Linear(latent_size,filters)
    self.activation = nn.Tanh()
    self.conv = nn.ModuleList()

    for in_channels, out_channels in zip((dense_channels, 128, 64, 32, 16), conv_filters):
      self.conv.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, padding='same'))

    
    # not used in the CNN model. It's a dummy variable used in PyGRANSO that has to be defined there, 
    # as PyGRANSO will read all parameters from the nn.parameters() 
    self.U =torch.nn.Parameter(torch.randn(50,50))

    # outputs = tf.squeeze(net, axis=[-1])

    # self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
    # self.z = self.add_weight(
    #     shape=inputs.shape, initializer=latent_initializer, name='z')


  def forward(self,x):
    x = self.dense(x)
    x = x.reshape((1,self.dense_channels,self.h, self.w))

    for resize, filters, i in zip(self.resizes, self.conv_filters,list(range(len(self.conv_filters)))):
      x = self.activation(x)
      # print(x.shape)
      x = UpSampling(resize)(x)
      # print(x.shape)
      x = normalization(x)
      x = self.conv[i](x)
      # print(x.shape)
      
    return x

      # TODO: weight intilizer for the convolution kernel
      # TODO: AddOffSet
      # TODO: squeeze output: [1,20,60,1] -> [1,20,60]  
      # TODO: latent initializer

      # if offset_scale != 0:
      #   net = AddOffset(offset_scale)(net)

