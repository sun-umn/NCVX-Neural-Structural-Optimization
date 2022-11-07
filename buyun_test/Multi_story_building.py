import sys
sys.path.append('/home/buyun/Documents/GitHub/NCVX-Neural-Structural-Optimization')
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import os
import gc

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

def constrained_structural_optimization_function(model, ke, args, designs, losses):
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
        device=device,
        dtype=double_precision
    )
    x_phys = torch.sigmoid(logits)

    # Calculate the forces
    forces = topo_physics.calculate_forces(x_phys, args)

    # Calculate the u_matrix
    u_matrix, _ = topo_physics.sparse_displace(
        x_phys, ke, forces, args["freedofs"], args["fixdofs"], **kwargs
    )

    # Calculate the compliance output
    compliance_output, _, _ = topo_physics.compliance(x_phys, u_matrix, ke, **kwargs)

    # The loss is the sum of the compliance
    f = torch.abs(torch.sum(compliance_output))

    # Run this problem with no inequality constraints
    ci = None

    # Run this problem with no equality constraints
    ce = pygransoStruct()
    ce.c1 = 1e3*(torch.mean(x_phys) - args['volfrac'])

    # Append updated physical density designs
    designs.append(
        x_phys
    )  # noqa

    return f, ci, ce

# Set devices and data type
n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    print("Use CUDA")
    gpu_list = ["cuda:{}".format(i) for i in range(n_gpu)]
    device = torch.device(gpu_list[0])
else:
    device = torch.device("cpu")

# device = torch.device("cpu")
double_precision = torch.double


# Identify the problem
# problem = problems.PROBLEMS_BY_NAME['multistory_building_32x64_0.5']
problem = problems.multistory_building(32, 64, density=0.5, device=device, dtype=double_precision)

# Get the problem args
args = topo_api.specified_task(problem, device=device, dtype=double_precision)
cnn_kwargs = None

# # Trials
# trials = []

# fix random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

for i in range(20):

    # Initialize the CNN Model
    if cnn_kwargs is not None:
        cnn_model = models.CNNModel(args, **cnn_kwargs).to(device=device, dtype=double_precision)
    else:
        cnn_model = models.CNNModel(args).to(device=device, dtype=double_precision)

    # Put the cnn model in training mode
    cnn_model.train()

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"],
        poisson=args["poisson"],
    ).to(device=device, dtype=double_precision)

    # Create the combined function and structural optimization
    # setup
    # Save the physical density designs & the losses
    designs = []
    losses = []
    # Combined function
    comb_fn = lambda model: constrained_structural_optimization_function(  # noqa
        model, ke, args, designs, losses
    )

    # Initalize the pygranso options
    opts = pygransoStruct()

    # Set the device
    # opts.torch_device = torch.device('cpu')
    opts.torch_device = device

    # Setup the intitial inputs for the solver
    nvar = getNvarTorch(cnn_model.parameters())
    opts.x0 = (
        torch.nn.utils.parameters_to_vector(cnn_model.parameters())
        .detach()
        .reshape(nvar, 1)
    ).to(device=device, dtype=double_precision)

    # Additional pygranso options
    opts.limited_mem_size = 20
    opts.double_precision = True
    opts.mu0 = 1.0
    opts.maxit = 500
    opts.print_frequency = 10
    opts.stat_l2_model = False
    opts.viol_eq_tol = 1e-8
    opts.opt_tol = 1e-8

    # Train pygranso
    start = time.time()
    soln = pygranso(var_spec=cnn_model, combined_fn=comb_fn, user_opts=opts)
    end = time.time()
    
    # Final structure
    pygranso_structure = designs[-1].detach().numpy()
    final_objective = soln.final.f
    
    # trials.append((final_objective, pygranso_structure))

    # best_trial = sorted(trials)[0]
    # best_trial[0]


    from scipy.ndimage import gaussian_filter

    pygranso_structure = gaussian_filter(pygranso_structure, sigma=1.2)
    # pygranso_structure = best_trial[1]

    # Plot the two structures together
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

    # pygranso
    ax1.imshow(pygranso_structure, cmap='Greys')
    ax1.grid()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Multi-story Building 32x64 Density 0.5')
    fig.tight_layout()
    fig.savefig("fig/multi-story-building/pygranso_test_xphys_{}.png".format(i))
    gc.collect()