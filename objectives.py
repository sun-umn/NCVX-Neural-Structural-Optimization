import torch
from neural_structural_optimization import topo_physics


def objective(x, ke, args, volume_constraint=True, filtering=False):
    """
    Objective function (compliance)
    for topology optimization.

    NOTE: U will be a parameter that we find via the objective
    """
    kwargs = dict(
        penal=torch.tensor(args["penal"]),
        e_min=torch.tensor(args["young_min"]),
        e_0=torch.tensor(args["young"]),
    )

    # Calculate the physical density based on the current values of x
    x_phys = topo_physics.physical_density(
        x, args, volume_constraint=volume_constraint, filtering=filtering
    )

    # Gather the forces for the problem
    forces = topo_physics.calculate_forces(x_phys, args)

    # Compute the u matrix
    u_matrix = topo_physics.displace(
        x_phys, ke, forces, args["freedofs"], args["fixdofs"], **kwargs
    )

    # print(f'U matrix {u_matrix.min()}')
    # print(f'x_phys {x_phys.sum()}')

    compliance_output = topo_physics.compliance(x_phys, u_matrix, ke, **kwargs)
    # print(f'Compliance sum value {torch.sum(compliance_output)}')

    # TODO: add logging to this loss function
    return torch.sum(compliance_output)
