import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import os
import pandas as pd

# import problems to solve
import problems
import experiments
import train
import topo_api
import topo_physics
import models
import utils

from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

from scipy.ndimage import gaussian_filter

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
        base="MATLAB",
    )
    x_phys = torch.sigmoid(logits)
    mask = torch.broadcast_to(args['mask'], x_phys.shape) > 0
    x_phys = x_phys * mask.int()

    # Calculate the forces
    forces = topo_physics.calculate_forces(x_phys, args)

    # Calculate the u_matrix
    u_matrix, _ = topo_physics.sparse_displace(
        x_phys, ke, args, forces, args["freedofs"], args["fixdofs"], **kwargs
    )

    # Calculate the compliance output
    compliance_output, _, _ = topo_physics.compliance(x_phys, u_matrix, ke, args, **kwargs)

    # The loss is the sum of the compliance
    f = torch.abs(torch.sum(compliance_output))

    # Run this problem with no inequality constraints
    ci = None

    # Run this problem with no equality constraints
    ce = pygransoStruct()
    ce.c1 = 1e3 * (torch.mean(x_phys[mask]) - args['volfrac'])

    # Append updated physical density designs
    designs.append(
        x_phys
    )  # noqa

    return f, ci, ce

# Identify the problem
# we really need more iterations to see the CNN-LBFGS method dominate
problem = problems.PROBLEMS_BY_NAME['thin_support_bridge_128x128_0.2']

# Get the arguments for the problem
args = topo_api.specified_task(problem)

# cnn_kwargs
cnn_kwargs = None

# Get the problem args
args = topo_api.specified_task(problem)

# Trials
trials = []

for seed in range(43, 44):
    torch.random.manual_seed(seed)

    # Initialize the CNN Model
    if cnn_kwargs is not None:
        cnn_model = models.CNNModel(args, **cnn_kwargs)
    else:
        cnn_model = models.CNNModel(args)

    # Put the cnn model in training mode
    cnn_model.train()

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
        model, ke, args, designs, losses
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
    opts.print_frequency = 10
    opts.stat_l2_model = False
    opts.viol_eq_tol = 1e-6
    opts.opt_tol = 1e-4
    
    mHLF_obj = utils.HaltLog()
    halt_log_fn, get_log_fn = mHLF_obj.makeHaltLogFunctions(opts.maxit)

    #  Set PyGRANSO's logging function in opts
    opts.halt_log_fn = halt_log_fn

    # Main algorithm with logging enabled.
    soln = pygranso(var_spec=cnn_model, combined_fn=comb_fn, user_opts=opts)

    # GET THE HISTORY OF ITERATES
    # Even if an error is thrown, the log generated until the error can be
    # obtained by calling get_log_fn()
    log = get_log_fn()
    
    # Final structure
    designs_indexes = (pd.Series(log.fn_evals).cumsum() - 1).values.tolist()
    final_designs = [designs[i] for i in designs_indexes]
    
    trials.append((soln.final.f, pd.Series(log.f), final_designs))

best_trial = sorted(trials)[0]

# Plot the loss
ax = best_trial[1].cummin().plot(figsize=(10, 7), marker='*')
ax.grid()

from scipy.ndimage import gaussian_filter

pygranso_structure = best_trial[2][-1].detach().numpy()

# Get the final frame
final_frame = pygranso_structure
reversed_frame = final_frame[:, ::-1]
bridge_frame = np.hstack([final_frame, reversed_frame] * 2)

# Create a figure and axis
fig, ax = plt.subplots(1, 1)

# Show the structure in grayscale
im = ax.imshow(bridge_frame, cmap='Greys')
ax.set_title('Thin Support Bridge')
ax.set_ylabel('Height')
ax.set_xlabel('Width')
ax.grid()
fig.colorbar(im, orientation="horizontal", pad=0.2)

fig.savefig("fig/bridge_test.png")