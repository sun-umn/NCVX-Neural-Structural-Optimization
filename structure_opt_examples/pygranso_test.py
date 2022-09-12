import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/NCVX-Experiments-PAMI')
# sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
import numpy as np
from utils import problems

device = torch.device('cuda')



problem = problems.mbb_beam(height=20, width=60)


# variables and corresponding dimensions.
var_in = {"V": [n,d]}

def user_fn(X_struct,A,d):
    V = X_struct.V

    # objective function
    f = -torch.trace(V.T@A@V)

    # inequality constraint, matrix form
    ci = None

    # equality constraint
    ce = pygransoStruct()
    ce.c1 = V.T@V - torch.eye(d).to(device=device, dtype=torch.double)

    return [f,ci,ce]

comb_fn = lambda X_struct : user_fn(X_struct,A,d)

opts = pygransoStruct()
opts.torch_device = device
opts.print_frequency = 1
# opts.opt_tol = 1e-7
opts.maxit = 3000
# opts.mu0 = 10
# opts.steering_c_viol = 0.02

start = time.time()
soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

V = torch.reshape(soln.final.x,(n,d))

rel_dist = torch.norm(V@V.T - U@U.T)/torch.norm(U@U.T)
print("torch.norm(V@V.T - U@U.T)/torch.norm(U@U.T) = {}".format(rel_dist))

print("torch.trace(V.T@A@V) = {}".format(torch.trace(V.T@A@V)))
print("torch.trace(U.T@A@U) = {}".format(torch.trace(U.T@A@U)))
print("sum of first d eigvals = {}".format(torch.sum(L[index[0:d]])))
print("sorted eigs = {}".format(L[index]))