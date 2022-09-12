"""A suite of topology optimization problems."""
from typing import Optional, Union
import dataclasses

import numpy as np

@dataclasses.dataclass
class Problem:
  """Description of a topology optimization problem.

  Attributes:
    normals: float64 array of shape (width+1, height+1, 2) where a value of 1
      indicates a "fixed" coordinate, and 0 indicates no normal force.
    forces: float64 array of shape (width+1, height+1, 2) indicating external
      applied forces in the x and y directions.
    density: fraction of the design region that should be non-zero.
    mask: scalar or float64 array of shape (height, width) that is multiplied by
      the design mask before and after applying the blurring filters. Values of
      1 indicate regions where the material can be optimized; values of 0 are
      constrained to be empty.
    name: optional name of this problem.
    width: integer width of the domain.
    height: integer height of the domain.
    mirror_left: should the design be mirrored to the left when displayed?
    mirror_right: should the design be mirrored to the right when displayed?
  """
  normals: np.ndarray
  forces: np.ndarray
  density: float
  mask: Union[np.ndarray, float] = 1
  name: Optional[str] = None
  width: int = dataclasses.field(init=False)
  height: int = dataclasses.field(init=False)
  mirror_left: bool = dataclasses.field(init=False)
  mirror_right: bool = dataclasses.field(init=False)

  def __post_init__(self):
    self.width = self.normals.shape[0] - 1
    self.height = self.normals.shape[1] - 1

    if self.normals.shape != (self.width + 1, self.height + 1, 2):
      raise ValueError(f'normals has wrong shape: {self.normals.shape}')
    if self.forces.shape != (self.width + 1, self.height + 1, 2):
      raise ValueError(f'forces has wrong shape: {self.forces.shape}')
    if (isinstance(self.mask, np.ndarray)
        and self.mask.shape != (self.height, self.width)):
      raise ValueError(f'mask has wrong shape: {self.mask.shape}')

    self.mirror_left = (
        self.normals[0, :, X].all() and not self.normals[0, :, Y].all()
    )
    self.mirror_right = (
        self.normals[-1, :, X].all() and not self.normals[-1, :, Y].all()
    )

def mbb_beam(width=60, height=20, density=0.5):
    """Textbook beam example."""
    X, Y = 0, 1

    width=60 
    height=20
    density=0.5

    normals = np.zeros((width + 1, height + 1, 2))
    normals[-1, -1, Y] = 1
    normals[0, :, X] = 1

    forces = np.zeros((width + 1, height + 1, 2))
    forces[0, 0, Y] = -1

    return Problem(normals, forces, density)