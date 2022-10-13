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
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch

n = 2562


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


F = torch.randn(n,1).to(device=device, dtype=double_precision)
K = torch.randn(n,n).to(device=device, dtype=double_precision)
u = torch.randn(n,1).to(device=device, dtype=double_precision)

# variables and corresponding dimensions.
var_in = {"K": [n,n], "u":[n,1]}

# # Put the model in training mode
# factorization_model.train()

def user_fn(X_struct,F,K,u):
    # K = X_struct.K
    u = X_struct.u
    # objective function
    f = 0
    f = torch.sum(torch.square(K@u-F))



    # inequality constraint
    ci = None
    # ci = pygransoStruct()
    # ci.c1 = torch.sum(torch.square(K@U-F))/n

    # equality constraint
    ce = pygransoStruct()
    # ce.c1 = torch.sum(torch.square(K@u-F))
    ce = None
    # print(ci.c1)
    # print(f)
    return [f,ci,ce]

comb_fn = lambda X_struct : user_fn(X_struct,F,K,u)

# PyGranso Options
opts = pygransoStruct()

# Set the device to CPU
opts.torch_device = device



# Additional PyGranso options
opts.limited_mem_size = 20
opts.double_precision = True
opts.mu0 = 1
opts.maxit = 5000
opts.print_frequency = 20
opts.stat_l2_model = False

start = time.time()
soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))