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

def structural_optimization_function(model,z,forces, ke, args, designs, kwargs, debug=False):
    """
    Combined function for PyGranso for the structural optimization
    problem. The inputs will be the model that reparameterizes x as a function
    of a neural network. V0 is the initial volume, K is the global stiffness
    matrix and F is the forces that are applied in the problem.
    """
    x_phys = torch.squeeze(model(z),1) # DIP like strategy

    u = list(model.parameters())[0] # dummy variable, shape: [2562,1]

    # feasible initialization
    with open('data_file/x_phys.npy', 'rb') as f:
        x_phys = torch.tensor(np.load(f)).to(device=kwargs['device'],dtype=kwargs['dtype'])

    with open('data_file/u_matrix.npy', 'rb') as f:
        u = torch.tensor(np.load(f)).to(device=kwargs['device'],dtype=kwargs['dtype'])

    dim_factor = u.shape[0]**0.5 # used for rescaling
    
    freedofs_forces,K, index_map = topo_physics.get_KU(
            x_phys, ke,forces, args['freedofs'], args['fixdofs'], **kwargs
        )

    # TODO: u_freedof[index_map] = u
    u_freedof = u[index_map][:len(args['freedofs'])]
    u_fixdof = u[index_map][len(args['freedofs']):]

    # test = torch.sum(( u_freedof )**2)**0.5

    # Calculate the compliance output u^T K u and force Ku
    compliance_output = topo_physics.compliance(x_phys, u, ke, **kwargs)

    # The loss is the sum of the compliance
    f = torch.sum(compliance_output)
    
    # inequality constraint, matrix form: x_phys\in[0,1]^d
    ci = pygransoStruct()
    box_constr = torch.hstack(
        (x_phys.reshape(x_phys.numel()) - 1,
        -x_phys.reshape(x_phys.numel()))
    )
    box_constr = torch.clamp(box_constr, min=0)
    folded_constr = torch.sum(box_constr**2)**0.5 #/dim_factor
    ci.c1 = folded_constr # folded 2562 constraints
    # ci = None

    # equality constraint
    ce = pygransoStruct()
    ce.c1 = torch.mean(x_phys) - args["volfrac"]
    ce.c2 = torch.sum((K.to_dense()@u_freedof - freedofs_forces)**2)**0.5 /dim_factor # folded 2562 constraints
    ce.c3 = torch.sum(u_fixdof**2)*0.5 /dim_factor
    # ce.c1 = torch.sum(K.to_dense()**2)**0.5


    # print("ci.c1 = {}, ce.c1 = {}, ce.c2 = {}, ce.c3 = {}".format(ci.c1,ce.c1,ce.c2, ce.c3))
    
    designs.append(x_phys)
    
    return [f, ci, ce]

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

# Identify the problem
problem = problems.mbb_beam(height=20, width=60, device=device, dtype=double_precision)

# Get the arguments for the problem
args = topo_api.specified_task(problem,device=device, dtype=double_precision)

# Set up the cnn args for this problem
cnn_kwargs = dict(resizes=(1, 1, 2, 2, 1))

# Initialize the model
cnn_model = models.CNNModel(
    args=args,
    **cnn_kwargs
).to(device=device, dtype=double_precision)

# Put the model in training mode
cnn_model.train()

fixed_random_input = torch.normal(mean=torch.zeros((1, 128)), std=torch.ones((1, 128))).to(device=device, dtype=double_precision)

# Calculate the forces (constant vector [2562,1])
forces = topo_physics.calculate_forces(x_phys=None, args=args)
freedofs_forces = forces[args['freedofs'].cpu().numpy()]

# # DEBUG part: print pygranso optimization variables
for name, param in cnn_model.named_parameters():
    print("{}: {}".format(name, param.data.shape))

# Get the stiffness matrix
ke = topo_physics.get_stiffness_matrix(
    young=args['young'], poisson=args['poisson'],device=device, dtype=double_precision
).to(device=device, dtype=double_precision)

# kwargs for displacement
kwargs = dict(
    penal=args["penal"],
    e_min=args["young_min"],
    e_0=args["young"],
    device=device,
    dtype=double_precision
)


# Structural optimization problem setup
designs = []
comb_fn = lambda model: structural_optimization_function(
    model, fixed_random_input, forces, ke, args, designs, kwargs, debug=False
)

# PyGranso Options
opts = pygransoStruct()

# Set the device to CPU
opts.torch_device = device

# Set up the initial inputs to the solver
nvar = getNvarTorch(cnn_model.parameters())
opts.x0 = (
    torch.nn.utils.parameters_to_vector(cnn_model.parameters())
    .detach()
    .reshape(nvar, 1)
).to(device=device, dtype=double_precision)

# Additional PyGranso options
opts.limited_mem_size = 20
opts.double_precision = True
opts.mu0 = 1e-5
opts.maxit = 1000
opts.print_frequency = 1
opts.stat_l2_model = False





# Run pygranso
start = time.time()
soln = pygranso(var_spec=cnn_model, combined_fn=comb_fn, user_opts=opts)
end = time.time()
print(f'Total wall time: {end - start}s')



# Get the final frame
final_frame = designs[-1].cpu().detach().numpy()

# Create a figure and axis
fig, ax = plt.subplots(1, 1)

# Show the structure in grayscale
im = ax.imshow(final_frame, cmap='Greys')
ax.set_title('MBB Beam - Neural Structural Optimization - PyGranso')
ax.set_ylabel('MBB Beam - Height')
ax.set_xlabel('MBB Beam - Width')
ax.grid()
fig.colorbar(im, orientation="horizontal", pad=0.2)
fig.savefig("fig/pygranso_test.png")