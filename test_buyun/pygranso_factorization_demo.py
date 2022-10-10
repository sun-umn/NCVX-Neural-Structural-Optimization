import torch.nn as nn
import torch.nn.functional as Fun
import torch
import numpy as np

# Import the models so we can follow the training code
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
import torch.optim as optim

print(torch.__version__)

import sys
sys.path.append('/home/buyun/Documents/GitHub/NCVX-Neural-Structural-Optimization')

# Topology Library
# import models
# import problems
# import topo_api
# import topo_physics

from str_opt_utils import models
from str_opt_utils import problems
from str_opt_utils import topo_api
from str_opt_utils import topo_physics



# first party
# Import pygranso functionality
# sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch

class nn_factorization(nn.Module):
  def __init__(self):
    super().__init__()
    self.U =torch.nn.Parameter(torch.randn(2562,1))
    self.K =torch.nn.Parameter(torch.randn(2562,2562))

  def forward(self):      
    return torch.sum(torch.square(self.K@self.U))**0.5


# Set devices and data type
n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    print("Use CUDA")
    gpu_list = ["cuda:{}".format(i) for i in range(n_gpu)]
    device = torch.device(gpu_list[0])
else:
    device = torch.device("cpu")

double_precision = torch.double

# fix random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize the model
factorization_model = nn_factorization().to(device=device, dtype=double_precision)

F = torch.randn(2562,1).to(device=device, dtype=double_precision)

# # DEBUG part: print pygranso optimization variables
for name, param in factorization_model.named_parameters():
    print("{}: {}".format(name, param.data.shape))

# # Put the model in training mode
# factorization_model.train()

def user_fn(model,F):
    # objective function
    f = model()*0

    U = list(model.parameters())[0]
    K = list(model.parameters())[1]

    # inequality constraint
    ci = None

    # equality constraint
    ce = pygransoStruct()
    ce.c1 = torch.sum(torch.square(K@U-F))**0.5

    return [f,ci,ce]

comb_fn = lambda model : user_fn(model,F)

# PyGranso Options
opts = pygransoStruct()

# Set the device to CPU
opts.torch_device = device

# Set up the initial inputs to the solver
nvar = getNvarTorch(factorization_model.parameters())
opts.x0 = (
    torch.nn.utils.parameters_to_vector(factorization_model.parameters())
    .detach()
    .reshape(nvar, 1)
).to(device=device, dtype=double_precision)

# Additional PyGranso options
opts.limited_mem_size = 20
opts.double_precision = True
opts.mu0 = 1
opts.maxit = 5000
opts.print_frequency = 30
opts.stat_l2_model = False

start = time.time()
soln = pygranso(var_spec= factorization_model, combined_fn = comb_fn, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))