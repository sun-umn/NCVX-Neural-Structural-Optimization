import time
import torch
import sys

# # from neural_structural_optimization.topo_physics import get_stiffness_matrix
# ## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/NCVX-Experiments-PAMI')
# sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import torch.nn as nn
import torch.nn.functional as Fun
import torch
import numpy as np

# from utils import models
# from utils import problems
# from utils import topo_api
# from utils import train

device = torch.device('cuda')

def normalization(inputs, epsilon=1e-6):
  variance, mean = torch.var_mean(inputs)
  x = inputs
  x -= mean
  x *= torch.rsqrt(variance + epsilon)
  return x

# def UpSampling(scale_factor):
#   return Fun.upsample(scale_factor = scale_factor, mode='bilinear') # AD problem with bilinear

class CNNModel_torch(nn.Module):

  def __init__(
      self,
    #   seed=0,
      args=None,
      latent_size=128,
      dense_channels=32,
    #   resizes=(1, 2, 2, 2, 1),
      conv_filters=(128, 64, 32, 16, 1),
    #   offset_scale=10,
      kernel_size=(5, 5),
    #   latent_scale=1.0,
      dense_init_scale=1.0,
    #   activation=nn.Tanh,
    #   conv_initializer=tf.initializers.VarianceScaling,
    #   normalization=global_normalization,
  ):
    # super().__init__(seed, args)
    super().__init__()
    resizes=(1, 1, 2, 2, 1)

    if len(resizes) != len(conv_filters):
      raise ValueError('resizes and filters must be same size')

    # self.activation = nn.Tanh

    total_resize = int(np.prod(resizes))
    self.h = h = 20 // total_resize
    self.w = w = 60 // total_resize
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
    # self.activation = nn.Tanh()
    self.conv = nn.ModuleList()

    for in_channels, out_channels in zip((dense_channels, 128, 64, 32, 16), conv_filters):
      self.conv.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, padding='same'))

    
    # not used in the CNN model. It's a dummy variable used in PyGRANSO that has to be defined there, 
    # as PyGRANSO will read all parameters from the nn.parameters() 
    self.U =torch.nn.Parameter(torch.randn(60*20,1))

    # outputs = tf.squeeze(net, axis=[-1])

    # self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
    # self.z = self.add_weight(
    #     shape=inputs.shape, initializer=latent_initializer, name='z')


  def forward(self,x):
    x = self.dense(x)
    x = x.reshape((1,self.dense_channels,self.h, self.w))

    for resize, filters, i in zip(self.resizes, self.conv_filters,list(range(len(self.conv_filters)))):
      x = Fun.tanh(x)
      # print(x.shape)
      x = Fun.upsample(x,scale_factor = resize, mode='bilinear')
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


def user_fn(model,z, V0, K, F):
    x = torch.squeeze(model(z),1)
    U = list(model.parameters())[0] # shape: [60*20,1]

    # objective function
    f = U.T@K@U

    # inequality constraint, matrix form
    ci = None

    # equality constraint
    ce = pygransoStruct()
    ce.c1 = torch.mean(x) - V0

    box_constr = torch.hstack(
        (x.reshape(x.numel()) - 1,
        -x.reshape(x.numel()))
    )
    box_constr = torch.clamp(box_constr, min=0)
    folded_constr = torch.sum(box_constr**2)**0.5
    ce.c2 = folded_constr
    ce.c3 = K@U - F

    return [f,ci,ce]

V0 = 0.5 # volume fraction from args
K = torch.randn((60*20,60*20)) # Stiffness matrix
F = torch.randn((60*20,1)) # Force vector
z = torch.randn(1,128) # initial fixed random input for DIP; similar to random seeds

model = CNNModel_torch()
model.train()



comb_fn = lambda model : user_fn(model,z, V0, K, F)



opts = pygransoStruct()
opts.torch_device = device
nvar = getNvarTorch(model.parameters())
opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
# opts.opt_tol = 1e-3
# opts.viol_eq_tol = 1e-4
# opts.maxit = 150
# opts.fvalquit = 1e-6
# opts.print_level = 1
# opts.print_frequency = 50
# opts.print_ascii = True
opts.limited_mem_size = 20
opts.double_precision = True

# opts.mu0 = 1


start = time.time()
soln = pygranso(var_spec= model, combined_fn = comb_fn, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))












# # DEBUG part:
# output = model(z)
# # [1,20,60] [batch_size,width,height]
# output = torch.squeeze(output,1)
# # print(list(model.parameters())[0].shape)
# for name, param in model.named_parameters():
#     print("{}: {}".format(name, param.data.shape))
