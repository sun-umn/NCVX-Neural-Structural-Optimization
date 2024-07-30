# stdlib
import os
import pickle
from typing import Any, Dict

# first party
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        1, 3, figsize=(12, 3), sharex=True, constrained_layout=True  # noqa
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


def _plot_design(
    design: np.ndarray,
    ax: matplotlib.axes._subplots.Subplot,
    loss: float,
    binary_constraint: float,
    volume_constraint: float,
    plot_configs: Dict[str, Any],
) -> None:
    """
    Function that creates the design plots
    """
    cmap = 'Greys'
    facecolor = plot_configs['facecolor']
    fontcolor = plot_configs['fontcolor']
    requires_flip = plot_configs['requires_flip']
    title = plot_configs['title']

    # Plot the design
    if requires_flip:
        design = np.hstack([design, design[:, ::-1]])

    ax.imshow(design, aspect='auto', cmap=cmap)
    ax.axis('off')
    ax.set_title(title, fontsize=14)

    # These methods are to nicely add the text. We can also
    # color code the face with the legend colors
    # Add the colors box for the scoring
    divider = make_axes_locatable(ax)

    cax = divider.append_axes("bottom", size="35%", pad=0.01)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)

    # Set the face color of the box
    cax.set_facecolor(facecolor)
    cax.spines["bottom"].set_color(facecolor)
    cax.spines["top"].set_color(facecolor)
    cax.spines["right"].set_color(facecolor)
    cax.spines["left"].set_color(facecolor)

    text = f"{loss} / {binary_constraint} / {volume_constraint}"
    cax.text(
        0.5,
        0.5,
        text,
        ha='center',
        va='center',
        fontsize=14,
        color=fontcolor,
    )


def build_designs(path: str, problem_name: str, experiment_id: str) -> None:
    """
    Function that is used to build the final designs of TO
    """
    # NTO-PCO Data
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

    # Initialize the subplots for the data
    # There are 4 methods total
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=(15, 1.5), constrained_layout=True
    )

    # NTO-PCO
    design = pygranso_data[0]
    loss = pygranso_data[1]
    binary_constraint = pygranso_data[2]
    volume_constraint = pygranso_data[3]

    requires_flip = False
    if 'bridge' in problem_name:
        requires_flip = True

    # Plot Configs
    pygranso_plot_configs = {
        'facecolor': 'blue',
        'fontcolor': 'white',
        'requires_flip': requires_flip,
        'title': 'NTO-PCO',
    }

    _plot_design(
        design=design,
        ax=ax1,
        loss=loss,
        binary_constraint=binary_constraint,
        volume_constraint=volume_constraint,
        plot_configs=pygranso_plot_configs,
    )

    # Google-DIP
    design = google_data['google-cnn'][0]
    loss = google_data['google-cnn'][1]
    binary_constraint = google_data['google-cnn'][2]
    volume_constraint = google_data['google-cnn'][3]

    # Plot Configs
    google_plot_configs = {
        'facecolor': 'red',
        'fontcolor': 'black',
        'requires_flip': requires_flip,
        'title': 'Google-DIP',
    }

    _plot_design(
        design=design,
        ax=ax2,
        loss=loss,
        binary_constraint=binary_constraint,
        volume_constraint=volume_constraint,
        plot_configs=google_plot_configs,
    )

    # TOuNN
    design = tounn_data[0]
    loss = tounn_data[1]
    binary_constraint = tounn_data[2]
    volume_constraint = tounn_data[3]

    tounn_plot_configs = {
        'facecolor': 'limegreen',
        'fontcolor': 'black',
        'requires_flip': requires_flip,
        'title': 'TOuNN',
    }

    _plot_design(
        design=design,
        ax=ax3,
        loss=loss,
        binary_constraint=binary_constraint,
        volume_constraint=volume_constraint,
        plot_configs=tounn_plot_configs,
    )

    # Classical Method MMA
    design = google_data['mma'][0]
    loss = google_data['mma'][1]
    binary_constraint = google_data['mma'][2]
    volume_constraint = google_data['mma'][3]

    mma_plot_configs = {
        'facecolor': 'gold',
        'fontcolor': 'black',
        'requires_flip': requires_flip,
        'title': 'MMA',
    }

    _plot_design(
        design=design,
        ax=ax4,
        loss=loss,
        binary_constraint=binary_constraint,
        volume_constraint=volume_constraint,
        plot_configs=mma_plot_configs,
    )

    img_filepath = os.path.join(path, f'{experiment_id}/{problem_name}-designs.png')
    fig.savefig(img_filepath, bbox_inches='tight')


def build_multi_model_size_results(
    path: str, experiment_config: dict, problem_name: str, title_mapping: dict
):
    """
    Function that computes the results for the multi-trial, multi-model
    experiments.

    experiments is a list that should represent [small, medium large]
    model experiment ids.
    """
    # Create data
    data = {}
    for model, experiment_id in experiment_config.items():
        data_path = os.path.join(
            path, f'{experiment_id}/{problem_name}-pygranso-cnn.pickle'
        )
        results = {}

        with open(data_path, 'rb') as f:
            # Final index contains all the relevant data
            model_data = pickle.load(f)[-1]

        # Loss, Binary Constraint, Volume Constraint
        results['loss'] = model_data['loss']
        results['binary_constraint'] = model_data['binary_constraint']
        results['volume_constraint'] = model_data['volume_constraint']

        # Add results data to data dictionary
        data[model] = results

    # Output will be a 3x3 result
    fig, axes = plt.subplots(
        3, 3, figsize=(11, 3.5), sharex=True, constrained_layout=True
    )

    # Colors
    colors = ['orange', 'blue', 'limegreen']

    # TODO: Currently hard-coded for our 20 trial experiments
    num_trials = 20
    for idx, (model, results) in enumerate(data.items()):
        # Print statistic
        trial_min_values = np.nanmin(results['loss'], axis=0)
        median_value = np.median(trial_min_values)
        min_value = np.min(trial_min_values)
        max_value = np.max(trial_min_values)

        print(f'{model} model loss; median={median_value}'.capitalize())
        print(f'{model} model loss; min={min_value}'.capitalize())
        print(f'{model} model loss; max={max_value}'.capitalize())
        print('\n')

        for result_idx, (key, values) in enumerate(results.items()):
            ax = axes[idx, result_idx]
            color = colors[idx]

            for trial in range(num_trials):
                values_as_series = pd.Series(np.abs(values[:, trial]))
                values_as_series.plot(color=color, logx=True, ax=ax)

            # Set titles along the x-axis
            if idx == 0 and result_idx == 0:
                ax.set_title('Compliance', fontsize=14)

            elif idx == 0 and result_idx == 1:
                ax.set_title('Binary Constraint', fontsize=14)

            elif idx == 0 and result_idx == 2:
                ax.set_title('Volume Constraint', fontsize=14)

            if result_idx == 1 or result_idx == 2:
                ax.text(425, 0.0, 'Goal', ha='left', va='bottom', color='Grey')
                ax.axhline(0.0, color='Grey')

            ax.grid()

    fig.suptitle(title_mapping[problem_name], fontsize=14)

    orange_patch = mpatches.Patch(color='orange', label='Small Model')
    blue_patch = mpatches.Patch(color='blue', label='Medium Model')
    green_patch = mpatches.Patch(color='limegreen', label='Large Model')

    fig.legend(
        handles=[orange_patch, blue_patch, green_patch],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
    )

    save_image_path = os.path.join(path, f'multi-size-model-{problem_name}-results.png')
    fig.savefig(save_image_path, bbox_inches='tight')
