# third party
import numpy as np
import torch


def specified_task(problem,device=torch.device('cpu'), dtype=torch.double):
    """
    Given a problem, return parameters for
    running a topology optimization.

    NOTE: Based on what I have been learning about pytorch
    we may need to update these inputs to be torch tensors.
    """
    fixdofs = np.flatnonzero(problem.normals.ravel().cpu().detach().clone())
    alldofs = np.arange(2 * (problem.width + 1) * (problem.height + 1))
    freedofs = np.sort(list(set(alldofs) - set(fixdofs)))

    params = {
        # material properties
        "young": torch.tensor(1),
        "young_min": torch.tensor(1e-9),
        "poisson": torch.tensor(0.3),
        "g": torch.tensor(0),
        # constraints
        "volfrac": torch.tensor(problem.density),
        "xmin": torch.tensor(0.001),
        "xmax": torch.tensor(1.0),
        # input parameters
        "nelx": torch.tensor(problem.width),
        "nely": torch.tensor(problem.height),
        "mask": torch.tensor(problem.mask).to(device=device, dtype=dtype),
        "freedofs": torch.tensor(freedofs).to(device=device, dtype=dtype),
        "fixdofs": torch.tensor(fixdofs).to(device=device, dtype=dtype),
        "forces": torch.tensor(problem.forces.ravel()),
        "penal": 3.0,
        "filter_width": 2,
    }
    return params
