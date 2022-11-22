# third party
import numpy as np
import torch

from utils import DEFAULT_DEVICE, DEFAULT_DTYPE


def specified_task(problem, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
    """
    Given a problem, return parameters for
    running a topology optimization.

    NOTE: Based on what I have been learning about pytorch
    we may need to update these inputs to be torch tensors.
    """
    fixdofs = np.flatnonzero(problem.normals.ravel().cpu().detach().clone())
    alldofs = np.arange(2 * (problem.width + 1) * (problem.height + 1))
    freedofs = np.sort(list(set(alldofs) - set(fixdofs)))

    # Variables that will utilize GPU calculations
    mask = torch.tensor(problem.mask).to(device=device, dtype=dtype)
    freedofs = torch.tensor(freedofs).to(device=device, dtype=torch.long)
    fixdofs = torch.tensor(fixdofs).to(device=device, dtype=torch.long)

    params = {
        # material properties
        "young": 1,
        "young_min": 1e-9,
        "poisson": 0.3,
        "g": 0,
        # constraints
        "volfrac": problem.density,
        "xmin": 0.001,
        "xmax": 1.0,
        # input parameters
        "nelx": torch.tensor(problem.width),
        "nely": torch.tensor(problem.height),
        "mask": mask,
        "freedofs": freedofs,
        "fixdofs": fixdofs,
        "forces": torch.tensor(problem.forces.ravel()),
        "penal": 3.0,
        "filter_width": 2,
    }
    return params
