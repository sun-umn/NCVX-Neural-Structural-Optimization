# Import the models so we can follow the training code
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('/home/buyun/Documents/GitHub/NCVX-Neural-Structural-Optimization')

# Topology Library
from str_opt_utils import problems
from str_opt_utils import train

# Identify the problem
problem = problems.mbb_beam(height=20, width=60)

# Set up the cnn args for this problem
cnn_kwargs = dict(resizes=(1, 1, 2, 2, 1))

rendered_frames, losses, x_phys, u_matrix = train.train_adam(problem, cnn_kwargs, lr=4e-3, iterations=50)

###################################################################
nely, nelx = x_phys.shape
ely, elx = np.meshgrid(range(nely), range(nelx))

# Nodes
n1 = (nely + 1) * (elx + 0) + (ely + 0)
n2 = (nely + 1) * (elx + 1) + (ely + 0)
n3 = (nely + 1) * (elx + 1) + (ely + 1)
n4 = (nely + 1) * (elx + 0) + (ely + 1)
all_ixs = np.array(
    [
        2 * n1,
        2 * n1 + 1,
        2 * n2,
        2 * n2 + 1,
        2 * n3,
        2 * n3 + 1,
        2 * n4,
        2 * n4 + 1,
    ]  # noqa
)
all_ixs = torch.from_numpy(all_ixs)

# The selected u and now needs to be multiplied by K
u_selected = u_matrix[all_ixs].squeeze()
###################################################################


# Get the final frame
final_frame = rendered_frames[-1].detach().numpy()

# Create a figure and axis
fig, ax = plt.subplots(1, 1)

# Show the structure in grayscale
im = ax.imshow(final_frame, cmap='Greys')
ax.set_title('MBB Beam - Neural Structural Optimization - Adam')
ax.set_ylabel('MBB Beam - Height')
ax.set_xlabel('MBB Beam - Width')
fig.colorbar(im, orientation="horizontal", pad=0.2)

fig.savefig("fig/cnn_test.png")

# with open('x_phys.npy','wb') as f:
#     np.save(f,x_phys.detach().cpu().numpy())

# with open('u_matrix.npy','wb') as f:
#     np.save(f,u_matrix.detach().cpu().numpy())