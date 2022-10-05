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

rendered_frames, losses = train.train_adam(problem, cnn_kwargs, lr=4e-3, iterations=500)


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