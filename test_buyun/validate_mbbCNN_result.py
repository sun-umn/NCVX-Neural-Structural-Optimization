import numpy as np
import torch

import sys
sys.path.append('/home/buyun/Documents/GitHub/NCVX-Neural-Structural-Optimization')

from str_opt_utils import topo_physics, topo_api, problems



with open('data_file/x_phys.npy', 'rb') as f:
    x_phys = torch.tensor(np.load(f))

with open('data_file/u_matrix.npy', 'rb') as f:
    u = torch.tensor(np.load(f))

# Identify the problem
problem = problems.mbb_beam(height=20, width=60)

# Get the arguments for the problem
args = topo_api.specified_task(problem)

# Get the stiffness matrix
ke = topo_physics.get_stiffness_matrix(
    young=args['young'], poisson=args['poisson']
)    

# Calculate the forces (constant vector [2562,1])
forces = topo_physics.calculate_forces(x_phys=None, args=args)

# kwargs for displacement
kwargs = dict(
    penal=args["penal"],
    e_min=args["young_min"],
    e_0=args["young"],
    device=torch.device("cpu"),
    dtype=torch.double
)

freedofs_forces, K, index_map = topo_physics.get_KU(
        x_phys, ke, forces, args['freedofs'], args['fixdofs'], **kwargs
    )

u_original = torch.zeros_like(u)
u_original[index_map] = u
# u_freedof = torch.tensor(u_original[:len(args['freedofs'])]).to(device=kwargs['device'],dtype=kwargs['dtype'])
u_freedof = u_original[:len(args['freedofs'])]
freedofs_forces = freedofs_forces.numpy()


# Calculate the compliance output u^T K u and force Ku
compliance_output = topo_physics.compliance(x_phys, u, ke, **kwargs)

# The loss is the sum of the compliance
f = torch.sum(compliance_output)

box_constr = torch.hstack(
        (x_phys.reshape(x_phys.numel()) - 1,
        -x_phys.reshape(x_phys.numel()))
    )
box_constr = torch.clamp(box_constr, min=0)
ci_c1 = torch.sum(box_constr**2)**0.5

ce_c1 = torch.mean(x_phys) - 0.5

# K = topo_physics.get_K(x_phys,ke,args,kwargs)

ce_c2 = np.sum((K@u_freedof - freedofs_forces)**2)**0.5

print("\n f = {}, folded constr x \in [0,1] = {}, mean(x_phys)-V0 = {}, F-KU = {} ".format(f.item(),ci_c1.item(),ce_c1.item(),ce_c2.item()))

print(1)