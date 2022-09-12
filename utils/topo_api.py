import autograd.numpy as np

def specified_task(problem):
  """Given a problem, return parameters for running a topology optimization."""
  fixdofs = np.flatnonzero(problem.normals.ravel())
  alldofs = np.arange(2 * (problem.width + 1) * (problem.height + 1))
  freedofs = np.sort(list(set(alldofs) - set(fixdofs)))

  params = {
      # material properties
      'young': 1,
      'young_min': 1e-9,
      'poisson': 0.3,
      'g': 0,
      # constraints
      'volfrac': problem.density,
      'xmin': 0.001,
      'xmax': 1.0,
      # input parameters
      'nelx': problem.width,
      'nely': problem.height,
      'mask': problem.mask,
      'freedofs': freedofs,
      'fixdofs': fixdofs,
      'forces': problem.forces.ravel(),
      'penal': 3.0,
      'filter_width': 2,
  }
  return params