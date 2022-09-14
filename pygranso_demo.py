import time
import torch
import sys
# Adding current directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/NCVX-Experiments-PAMI')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch

from utils import models

device = torch.device('cuda')
double_precision = torch.double
torch.manual_seed(42)

def user_fn(model,z, V0, K, F):
    x = torch.squeeze(model(z),1) # DIP like strategy
    U = list(model.parameters())[0] # dummy variable, shape: [60*20,1]

    # objective function
    f = U.T@K@U

    # inequality constraint, matrix form
    ci = pygransoStruct()
    box_constr = torch.hstack(
        (x.reshape(x.numel()) - 1,
        -x.reshape(x.numel()))
    )
    box_constr = torch.clamp(box_constr, min=0)
    folded_constr = torch.sum(box_constr**2)**0.5
    ci.c1 = folded_constr

    # equality constraint
    ce = pygransoStruct()
    ce.c1 = torch.mean(x) - V0
    ce.c2 = torch.sum((K@U - F)**2)**0.5 

    return [f,ci,ce]

V0 = 0.5 # volume fraction from args
K = torch.randn((60*20,60*20)).to(device=device, dtype=double_precision) # Stiffness matrix
F = torch.randn((60*20,1)).to(device=device, dtype=double_precision) # Force vector
z = torch.randn(1,128).to(device=device, dtype=double_precision) # initial fixed random input for DIP; similar to random seeds

model = models.CNNModel_torch().to(device=device, dtype=double_precision)
model.train()

comb_fn = lambda model : user_fn(model,z, V0, K, F)

# PyGRANSO options
opts = pygransoStruct()
opts.torch_device = device
nvar = getNvarTorch(model.parameters())
opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
opts.limited_mem_size = 20
opts.double_precision = True
opts.mu0 = 1
opts.maxit = 200

# torch.autograd.set_detect_anomaly(True) # DEBUG option

# # DEBUG part: print pygranso optimization variables
# for name, param in model.named_parameters():
#     print("{}: {}".format(name, param.data.shape))

# main function call
start = time.time()
soln = pygranso(var_spec= model, combined_fn = comb_fn, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))






