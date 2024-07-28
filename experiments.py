# stdlib
import os
import pickle
from typing import Dict

# first party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_optimization_trajectories(
    path: str,
    problem_name: str,
    experiment_id: str,
    title_mapping: Dict,
    font_size: int = 14,
) -> None:
    """
    Function that is used to build the optimization trajectories of
    Topology Optimization
    """
    # Create a figure
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(12, 3), sharex=True, constrained_layout=True
    )

    # PyGranso Data
    pygranso_data_path = os.path.join(
        path, f'{experiment_id}/{problem_name}-pygranso-cnn.pickle'
    )
    with open(pygranso_data_path, 'rb') as f:
        pygranso_data = pickle.load(f)

    # Google Data
    google_data_path = os.path.join(
        path, f'{experiment_id}/{problem_name}-google.pickle'
    )
    with open(google_data_path, 'rb') as f:
        google_data = pickle.load(f)

    # TOuNN Data
    tounn_data_path = os.path.join(path, f'{experiment_id}/{problem_name}-tounn.pickle')
    with open(tounn_data_path, 'rb') as f:
        tounn_data = pickle.load(f)

    # The last element for each of the data sources is the dictionary of
    # training trajectories
    # NTO-PCO Loss
    pygranso_data_trajectories = pygranso_data[-1]
    pygranso_loss = pd.Series(pygranso_data_trajectories['loss'].flatten())

    # For this function let's also print out the number of iterations for
    # convergence
    print(f'Our method converged in {len(pygranso_loss.dropna())} iterations')

    # Google-DIP Loss
    google_data_trajectories = google_data['google-cnn'][-1]
    google_loss = pd.Series(google_data_trajectories['loss'].values)

    # MMA Loss
    mma_data_trajectories = google_data['mma'][-1]
    mma_loss = pd.Series(mma_data_trajectories['loss'].values)

    # TOuNN Loss
    tounn_data_trajectories = tounn_data[-1]
    tounn_loss = pd.Series(tounn_data_trajectories['loss'])

    # Plot the losses
    # Compliance and iterations can be large so we are plotting with a log scale
    pygranso_loss.plot(color='blue', logx=True, logy=True, label='NTO-PCO', ax=ax1)
    google_loss.plot(color='red', label='Google-DIP', ax=ax1)
    mma_loss.plot(color='gold', label='MMA', ax=ax1)
    tounn_loss.plot(color='limegreen', label='TOuNN', ax=ax1)

    # Plot configs
    ax1.grid()
    ax1.set_title('Log (Compliance)', fontsize=font_size)

    # NTO-PCO Binary Constraint
    pygranso_bc = np.abs(
        pd.Series(pygranso_data_trajectories['binary_constraint'].flatten())
    )

    # Google-DIP and MMA Binary Constraint
    google_bc = np.abs(pd.Series(google_data_trajectories['binary_constraint']))
    mma_bc = np.abs(pd.Series(mma_data_trajectories['binary_constraint']))

    # TOuNN Binary Constraint
    tounn_data_trajectories = tounn_data[-1]
    tounn_bc = np.abs(pd.Series(tounn_data_trajectories['binary_constraint']))

    # Plot Binary Constraints
    pygranso_bc.plot(color='blue', logx=True, label='NTO-PCO', ax=ax2)
    google_bc.plot(color='red', label='Google-DIP', ax=ax2)
    mma_bc.plot(color='gold', label='MMA', ax=ax2)
    tounn_bc.plot(color='limegreen', label='TOuNN', ax=ax2)

    # Plot Configs
    ax2.grid()
    ax2.set_title('Binary Constraint', fontsize=font_size)
    ax2.set_xlabel('Log (Iterations)', fontsize=font_size)

    # To emphasize the goal of the paper - lets add a goal line
    # for the constraints
    ax2.text(0.75, 0.0, 'Goal', ha='left', va='bottom', color='gray')
    ax2.axhline(0.0, color='gray')

    # NTO-PCO Volume Constraint
    pygranso_vc = np.abs(
        pd.Series(pygranso_data_trajectories['volume_constraint'].flatten())
    )

    # Google-DIP and MMA Volume Constraint
    google_vc = np.abs(pd.Series(google_data_trajectories['volume_constraint']))
    mma_vc = np.abs(pd.Series(mma_data_trajectories['volume_constraint']))

    # TOuNN Volume Constraint
    tounn_vc = np.abs(pd.Series(tounn_data_trajectories['volume_constraint']))

    # Plot Volume Constraints
    pygranso_vc.plot(color='blue', logx=True, label='NTO-PCO', ax=ax3)
    google_vc.plot(color='red', label='Google-DIP', ax=ax3)
    mma_vc.plot(color='gold', label='MMA', ax=ax3)
    tounn_vc.plot(color='limegreen', label='TOuNN', ax=ax3)

    # Plot configs
    ax3.grid()
    ax3.set_title('Volume Constraint', fontsize=font_size)

    # Similar to the binary constraint I am also adding a goal for the
    # volume constraint
    ax3.text(0.75, 0.0, 'Goal', ha='left', va='bottom', color='gray')
    ax3.axhline(0.0, color='gray')

    # Single legend defined at outside the plot
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)

    # Add title and save the final outputs
    title = title_mapping[problem_name]
    fig.suptitle(title, fontsize=16)
    img_filepath = os.path.join(path, f'{experiment_id}/{problem_name}-performance.png')
    fig.savefig(img_filepath, bbox_inches='tight')
