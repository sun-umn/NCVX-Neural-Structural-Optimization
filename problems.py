# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A suite of topology optimization problems."""
# third party
import dataclasses
from typing import Optional, Union

import numpy as np
import torch

from utils import DEFAULT_DEVICE, DEFAULT_DTYPE

X, Y = 0, 1


@dataclasses.dataclass
class Problem:
    """
    Description of a topology optimization problem.
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

    normals: torch.Tensor
    forces: torch.Tensor
    density: float
    mask: Union[torch.Tensor, float] = 1
    name: Optional[str] = None
    width: int = dataclasses.field(init=False)
    height: int = dataclasses.field(init=False)
    mirror_left: bool = dataclasses.field(init=False)
    mirror_right: bool = dataclasses.field(init=False)

    def __post_init__(self):  # noqa
        self.width = self.normals.shape[0] - 1
        self.height = self.normals.shape[1] - 1

        if self.normals.shape != (self.width + 1, self.height + 1, 2):
            raise ValueError(f"normals has wrong shape: {self.normals.shape}")
        if self.forces.shape != (self.width + 1, self.height + 1, 2):
            raise ValueError(f"forces has wrong shape: {self.forces.shape}")
        if isinstance(self.mask, torch.Tensor) and self.mask.shape != (
            self.height,
            self.width,
        ):
            raise ValueError(f"mask has wrong shape: {self.mask.shape}")

        self.mirror_left = (
            self.normals[0, :, X].all() and not self.normals[0, :, Y].all()
        )
        self.mirror_right = (
            self.normals[-1, :, X].all() and not self.normals[-1, :, Y].all()
        )


def mbb_beam(
    width=60, height=20, density=0.5, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE
):
    """Textbook beam example."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, -1, Y] = 1
    normals[0, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[0, 0, Y] = -1

    return Problem(normals, forces, density)


def cantilever_beam_full(
    width=60,
    height=60,
    density=0.5,
    force_position=0,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Cantilever supported everywhere on the left"""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[0, :, :] = 1.0

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[-1, round((1 - force_position) * height), Y] = -1

    return Problem(normals, forces, density)


def cantilever_beam_two_point(
    width=60,
    height=60,
    density=0.5,
    support_position=0.25,
    force_position=0.5,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Cantilever supported by two points"""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[0, round(height * (1 - support_position)), :] = 1
    normals[0, round(height * support_position), :] = 1

    forces = torch.zeros((width + 1, height + 1, 2))
    forces[-1, round((1 - force_position) * height), Y] = -1

    return Problem(normals, forces, density)


def pure_bending_moment(
    width=60,
    height=60,
    density=0.5,
    support_position=0.45,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """Pure bending forces on a beam."""
    # Figure 28 from
    # http://naca.central.cranfield.ac.uk/reports/arc/rm/3303.pdf
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[-1, :, X] = 1

    # for numerical stability, fix y forces here at 0
    normals[0, round(height * (1 - support_position)), Y] = 1
    normals[0, round(height * support_position), Y] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[0, round(height * (1 - support_position)), X] = 1
    forces[0, round(height * support_position), X] = -1

    return Problem(normals, forces, density)


def multistory_building(
    width=32,
    height=32,
    density=0.3,
    interval=16,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """A multi-story building, supported from the ground."""
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[:, -1, Y] = 1
    normals[-1, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[:, ::interval, Y] = -1 / width
    return Problem(normals, forces, density)


def thin_support_bridge(
    width=32,
    height=32,
    density=0.25,
    design_width=0.25,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
):
    """
    A bridge supported from below with fixed width supports.
    """
    normals = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    normals[:, -1, Y] = 1
    normals[0, :, X] = 1
    normals[-1, :, X] = 1

    forces = torch.zeros((width + 1, height + 1, 2)).to(device=device, dtype=dtype)
    forces[:, 0, Y] = -1 / width

    mask = torch.ones((width, height)).to(device=device, dtype=dtype)
    mask[-round(width * (1 - design_width)) :, : round(height * (1 - design_width))] = 0

    return Problem(normals, forces, density, mask)


# Problems Category
PROBLEMS_BY_CATEGORY = {
    # idealized beam and cantilevers
    "mbb_beam": [
        mbb_beam(96, 32, density=0.5),
        mbb_beam(192, 64, density=0.4),
        mbb_beam(384, 128, density=0.3),
        mbb_beam(192, 32, density=0.5),
        mbb_beam(384, 64, density=0.4),
    ],
    "cantilever_beam_full": [
        cantilever_beam_full(96, 32, density=0.5),
        cantilever_beam_full(192, 64, density=0.3),
        cantilever_beam_full(384, 128, density=0.2),
        cantilever_beam_full(384, 128, density=0.15),
    ],
    "cantilever_beam_two_point": [
        cantilever_beam_two_point(64, 48, density=0.4),
        cantilever_beam_two_point(128, 96, density=0.3),
        cantilever_beam_two_point(256, 192, density=0.2),
        cantilever_beam_two_point(256, 192, density=0.15),
    ],
    "pure_bending_moment": [
        pure_bending_moment(32, 64, density=0.15),
        pure_bending_moment(64, 128, density=0.125),
        pure_bending_moment(128, 256, density=0.1),
    ],
    "multistory_building": [
        multistory_building(32, 64, density=0.5),
        multistory_building(64, 128, interval=32, density=0.4),
        multistory_building(128, 256, interval=64, density=0.3),
        multistory_building(128, 512, interval=64, density=0.25),
        multistory_building(128, 512, interval=128, density=0.2),
    ],
    "thin_support_bridge": [
        thin_support_bridge(64, 64, density=0.3),
        thin_support_bridge(128, 128, density=0.2),
        thin_support_bridge(256, 256, density=0.15),
        thin_support_bridge(256, 256, density=0.1),
    ],
}


# Create the ability to do problems by name the same as the paper
PROBLEMS_BY_NAME = {}
for problem_class, problem_list in PROBLEMS_BY_CATEGORY.items():
    for problem in problem_list:
        name = f"{problem_class}_{problem.width}x{problem.height}_{problem.density}"
        problem.name = name
        assert name not in PROBLEMS_BY_NAME, f"redundant name {name}"
        PROBLEMS_BY_NAME[name] = problem
