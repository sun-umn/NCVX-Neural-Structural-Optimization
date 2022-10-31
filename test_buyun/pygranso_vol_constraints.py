# Import the models so we can follow the training code
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

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



def structural_optimization_function(model,z,forces, ke, args, designs, displacement_frames, kwargs):
    """
    Combined function for PyGranso for the structural optimization
    problem. The inputs will be the model that reparameterizes x as a function
    of a neural network. V0 is the initial volume, K is the global stiffness
    matrix and F is the forces that are applied in the problem.
    """

    logits = model(None)
    x_phys = torch.sigmoid(logits)

    # Calculate the forces
    forces = topo_physics.calculate_forces(x_phys, args)

    # Calculate the u_matrix
    u_matrix, _ = topo_physics.sparse_displace(
        x_phys, ke, forces, args["freedofs"], args["fixdofs"], **kwargs
    )

    dim_factor = 2540 #u.shape[0]#**0.5 # used for rescaling




    # The loss is the sum of the compliance
    # Calculate the compliance output u^T K u and force Ku
    compliance_output = topo_physics.compliance(x_phys, u_matrix, ke, **kwargs)
    f = torch.sum(compliance_output)
    # f = torch.sum(u@freedofs_forces)*f_factor + (torch.mean(x_phys) - args["volfrac"])*0 
    # f = torch.sum(u@K.to_dense()@u)*f_factor
    
    # inequality constraint
    ci = None

    
    # equality constraint
    ce = pygransoStruct()
    ce.c1 = torch.abs(torch.mean(x_phys) - args['volfrac'])*1e4


    if len(designs) == 0:
        designs.append(x_phys)
    else:
        designs[0] = x_phys
        
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
cnn_model = models.CNNModel_xphys(
    args=args,
    **cnn_kwargs
).to(device=device, dtype=double_precision)

# Put the model in training mode
cnn_model.train()

fixed_random_input = torch.normal(mean=torch.zeros((1, 128)), std=torch.ones((1, 128))).to(device=device, dtype=double_precision)

# Calculate the forces (constant vector [2562,1])
forces = topo_physics.calculate_forces(x_phys=None, args=args)
# freedofs_forces = forces[args['freedofs'].cpu().numpy()]
# fixdofs_forces = forces[args['fixdofs'].cpu().numpy()]





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
)

# Additional PyGranso options
opts.limited_mem_size = 20
opts.double_precision = True
opts.mu0 = 1
opts.maxit = 150
opts.print_frequency = 1
opts.stat_l2_model = False
# opts.viol_ineq_tol = 1e-4
opts.viol_eq_tol = 1e-8
# opts.opt_tol = 1e-4

opts.init_step_size = 5e-5
opts.linesearch_maxit = 50
opts.linesearch_reattempts = 15


# Structural optimization problem setup
designs = []
displacement_frames = []



# Run pygranso
start = time.time()



comb_fn = lambda model: structural_optimization_function(
    model, fixed_random_input, forces, ke, args, designs,displacement_frames, kwargs
)

soln = pygranso(var_spec=cnn_model, combined_fn=comb_fn, user_opts=opts)


# # Get the final frame
final_frame = designs[-1].cpu().detach().numpy()

# final_frame = soln.final.x[2562:2562+1200].reshape(20,60).cpu().detach().numpy() # x_phys debug
# final_frame = torch.squeeze(model(fixed_random_input),1) # TODO: print final result for nn

# Create a figure and axis
fig, ax = plt.subplots(1, 1)

# Show the structure in grayscale
im = ax.imshow(final_frame, cmap='Greys')
ax.set_title('MBB Beam - Neural Structural Optimization - PyGranso')
ax.set_ylabel('MBB Beam - Height')
ax.set_xlabel('MBB Beam - Width')
ax.grid()
fig.colorbar(im, orientation="horizontal", pad=0.2)
fig.savefig("fig/pygranso_test_xphys.png")

    # final_displacement = displacement_frames[-1].cpu().detach().numpy()
    # # There will be 8 frames for the displacement field
    # # Create a figure and axis
    # fig, axes = plt.subplots(4, 2, figsize=(10, 9))
    # axes = axes.flatten()

    # # Go through the displacement fields
    # for index in range(final_displacement.shape[0]):
    #     displacement_image = final_displacement[index, :, :].T

    #     # Show the structure in grayscale
    #     axes[index].imshow(displacement_image,cmap="Greys")

    #     # if final_frame is not None:
    #     #     axes[index].imshow(final_frame, alpha=0.1, cmap="Greys")
    #     axes[index].set_title(f"Displacement Field {index + 1} Node")

    # fig.suptitle("Displacement Fields")
    # fig.tight_layout()
    # fig.colorbar(im, orientation="horizontal", pad=0.2)
    # fig.savefig("fig/pygranso_test_displacement.png")

end = time.time()
print(f'Total wall time: {end - start}s')



# # # Get the final frame
# final_frame = designs[-1].cpu().detach().numpy()

# # final_frame = soln.final.x[2562:2562+1200].reshape(20,60).cpu().detach().numpy() # x_phys debug
# # final_frame = torch.squeeze(model(fixed_random_input),1) # TODO: print final result for nn

# # Create a figure and axis
# fig, ax = plt.subplots(1, 1)

# # Show the structure in grayscale
# im = ax.imshow(final_frame, cmap='Greys')
# ax.set_title('MBB Beam - Neural Structural Optimization - PyGranso')
# ax.set_ylabel('MBB Beam - Height')
# ax.set_xlabel('MBB Beam - Width')
# ax.grid()
# fig.colorbar(im, orientation="horizontal", pad=0.2)
# fig.savefig("fig/pygranso_test.png")