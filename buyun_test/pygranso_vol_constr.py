import sys
sys.path.append('/home/buyun/Documents/GitHub/NCVX-Neural-Structural-Optimization')
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import os

# import problems to solve
import problems
import experiments
import train
import topo_api
import topo_physics
import models

from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

import torch.nn.functional as Fun


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return Fun.hardtanh(grad_output)

def constrained_structural_optimization_function(model, z, ke, args, designs, losses):
    """
    Combined function for PyGranso for the structural optimization
    problem. The inputs will be the model that reparameterizes x as a function
    of a neural network. V0 is the initial volume, K is the global stiffness
    matrix and F is the forces that are applied in the problem.
    """
    # Initialize the model
    # In my version of the model it follows the similar behavior of the
    # tensorflow repository and only needs None to initialize and output
    # a first value of x
    logits = model(None)

    # kwargs for displacement
    kwargs = dict(
        penal=torch.tensor(args["penal"]),
        e_min=torch.tensor(args["young_min"]),
        e_0=torch.tensor(args["young"]),
    )

    x_phys = torch.sigmoid(logits)

    # x_phys = STEFunction.apply(torch.sigmoid(logits))

    # Calculate the forces
    forces = topo_physics.calculate_forces(x_phys, args)

    # Calculate the u_matrix
    u_matrix, _ = topo_physics.sparse_displace(
        x_phys, ke, forces, args["freedofs"], args["fixdofs"], **kwargs
    )

    # Calculate the compliance output
    compliance_output,_,_ = topo_physics.compliance(x_phys, u_matrix, ke, **kwargs)

    # The loss is the sum of the compliance
    f = torch.abs(torch.sum(compliance_output)) #+ 1e4 * (torch.mean(x_phys) - args['volfrac'])**2

    # Run this problem with no inequality constraints
    ci = None

    # Run this problem with no equality constraints
    ce = pygransoStruct()
    ce.c1 = 1e4 * (torch.mean(x_phys) - args['volfrac'])
    # ce.c2 = torch.linalg.norm((x_phys**2-x_phys),ord=2)**2

    # Append updated physical density designs
    designs.append(
        x_phys
    )  # noqa

    return f, ci, ce

# fix random seed
seed = 43
torch.manual_seed(seed)
np.random.seed(seed)


# Identify the problem
problem = problems.mbb_beam(height=20, width=60)
problem.name = 'mbb_beam'

# cnn_kwargs
cnn_kwargs = dict(resizes=(1, 1, 2, 2, 1))

# Get the problem args
args = topo_api.specified_task(problem)

# Initialize the CNN Model
if cnn_kwargs is not None:
    cnn_model = models.CNNModel(args, **cnn_kwargs)
else:
    cnn_model = models.CNNModel(args)

# Put the cnn model in training mode
cnn_model.train()

fixed_random_input = torch.normal(mean=torch.zeros((1, 128)), std=torch.ones((1, 128)))#.to(device=device, dtype=double_precision)

# Create the stiffness matrix
ke = topo_physics.get_stiffness_matrix(
    young=args["young"],
    poisson=args["poisson"],
)

# Create the combined function and structural optimization
# setup
# Save the physical density designs & the losses
designs = []
losses = []
# Combined function
comb_fn = lambda model: constrained_structural_optimization_function(  # noqa
    model, fixed_random_input, ke, args, designs, losses
)

# Initalize the pygranso options
opts = pygransoStruct()

# Set the device
opts.torch_device = torch.device('cpu')

# Setup the intitial inputs for the solver
nvar = getNvarTorch(cnn_model.parameters())
opts.x0 = (
    torch.nn.utils.parameters_to_vector(cnn_model.parameters())
    .detach()
    .reshape(nvar, 1)
)

# Additional pygranso options
opts.limited_mem_size = 20
opts.double_precision = True
opts.mu0 = 1.0
opts.maxit = 500
# opts.print_frequency = 10
opts.stat_l2_model = False

opts.viol_ineq_tol = 1e-6
opts.viol_eq_tol = 1e-6
opts.opt_tol = 1e-6

# Other parameters that helped the structural optimization
# problem
# opts.init_step_size = 5e-5
# opts.linesearch_maxit = 50
# opts.linesearch_reattempts = 15

# Train pygranso
start = time.time()
soln = pygranso(var_spec=cnn_model, combined_fn=comb_fn, user_opts=opts)
end = time.time()
print(f'Total wall time: {end - start}s')

pygranso_structure = designs[-1].detach().numpy()

# Plot the two structures together
fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

# pygranso
ax1.imshow(pygranso_structure, cmap='Greys')
ax1.grid()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('MBB Beam 60x20 - Neural Structural Optimization')
fig.tight_layout()
fig.savefig("fig/pygranso_test_xphys.png")
