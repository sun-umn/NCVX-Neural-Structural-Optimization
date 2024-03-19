#!/usr/bin/python
# stdlib
import gc
import os
import warnings

# third party
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import xarray
from mpl_toolkits.axes_grid1 import make_axes_locatable
from neural_structural_optimization import models as google_models
from neural_structural_optimization import topo_api as google_topo_api
from neural_structural_optimization import train as google_train

# first party
import models
import problems
import topo_api
import train
import utils
from TOuNN.TOuNN import TopologyOptimizer

# Filter warnings
warnings.filterwarnings('ignore')

# NOTES:
# Keep these imports
# from matplotlib.offsetbox import AnchoredText
# from mpl_toolkits.axes_grid1 import make_axes_locatable


# Define the cli group
@click.group()
def cli():  # noqa
    pass


def calculate_binary_constraint(design, mask, epsilon):
    """
    Function to compute the binary constraint
    """
    return np.round(np.mean(design[mask] * (1 - design[mask])) - epsilon, 4)


def calculate_volume_constraint(design, mask, volume):
    """
    Function that computes the volume constraint
    """
    return np.round(np.mean(design[mask]) / volume - 1.0, 4)


def build_outputs(problem_name, outputs, mask, volume, requires_flip, epsilon=1e-3):
    """
    From each of the methods we will have an outputs
    based on the number of trials. This function
    will put together the outputs of the best trial.

    Outputs is a Dict object and should have keys:
    1. designs
    2. losses
    3. volumes
    4. binary_constraint
    5. trials_initial_volumnes
    """
    # Get the losses and sort from lowest to highest
    losses_df = pd.DataFrame(outputs["losses"])
    losses_df = losses_df.ffill()

    # Sort the final losses by index
    losses = np.min(losses_df, axis=0).values
    losses_indexes = np.argsort(losses)

    # Reorder outputs
    # losses
    losses_df = losses_df.iloc[:, losses_indexes]

    # final designs
    final_designs = outputs["designs"]
    final_designs = final_designs[losses_indexes, :, :]

    # Get all final objects
    best_final_design = final_designs[0, :, :]
    # Compute the binary and volume constraints
    binary_constraint = calculate_binary_constraint(
        design=best_final_design,
        mask=mask,
        epsilon=epsilon,
    )

    # volume constraint
    volume_constraint = calculate_volume_constraint(
        design=best_final_design,
        mask=mask,
        volume=volume,
    )

    if requires_flip:
        if (
            ("mbb" in problem_name)
            or ("l_shape" in problem_name)
            or ("cantilever" in problem_name)
        ):
            best_final_design = np.hstack(
                [best_final_design[:, ::-1], best_final_design]
            )

        elif (
            ("multistory" in problem_name)
            or ("thin" in problem_name)
            or ("michell" in problem_name)
        ):
            best_final_design = np.hstack(
                [best_final_design, best_final_design[:, ::-1]] * 2
            )

    # last row, first column (-1, 0)
    best_score = np.round(losses_df.iloc[-1, 0], 2)

    return best_final_design, best_score, binary_constraint, volume_constraint


def build_google_outputs(problem_name, ds, mask, volume, requires_flip, epsilon=1e-3):
    """
    Build the google outputs.
    TODO: I think I will want to extend this for multiple
    trials but will leave as a single trial for now
    """
    # Get the minimum loss for the benchmark methods
    losses = ds.loss.transpose().to_pandas().cummin().ffill()
    cnn_loss = np.round(losses["cnn-lbfgs"].min(), 2)
    mma_loss = np.round(losses["mma"].min(), 2)

    # Select the final step from the xarray
    final_designs = ds.design.sel(step=200, method="nearest").data

    # CNN final design
    cnn_final_design = final_designs[0, :, :]
    cnn_binary_constraint = calculate_binary_constraint(
        design=cnn_final_design, mask=mask, epsilon=epsilon
    )
    cnn_volume_constraint = calculate_volume_constraint(
        design=cnn_final_design,
        mask=mask,
        volume=volume,
    )

    # MMA final design
    mma_final_design = final_designs[1, :, :]
    mma_binary_constraint = calculate_binary_constraint(
        design=mma_final_design,
        mask=mask,
        epsilon=epsilon,
    )
    mma_volume_constraint = calculate_volume_constraint(
        design=mma_final_design,
        mask=mask,
        volume=volume,
    )

    if requires_flip:
        if ("mbb" in problem_name) or ("cantilever" in problem_name):
            cnn_final_design = np.hstack([cnn_final_design[:, ::-1], cnn_final_design])
            mma_final_design = np.hstack([mma_final_design[:, ::-1], mma_final_design])

        if (
            ("multistory" in problem_name)
            or ("thin" in problem_name)
            or ("michell" in problem_name)
        ):
            cnn_final_design = np.hstack(
                [cnn_final_design, cnn_final_design[:, ::-1]] * 2
            )
            mma_final_design = np.hstack(
                [mma_final_design, mma_final_design[:, ::-1]] * 2
            )

    return {
        "google-cnn": (
            cnn_final_design,
            cnn_loss,
            cnn_binary_constraint,
            cnn_volume_constraint,
        ),
        "mma": (
            mma_final_design,
            mma_loss,
            mma_binary_constraint,
            mma_volume_constraint,
        ),
    }


def train_all(problem, max_iterations, cnn_kwargs=None):
    """
    Function that will compute the MMA and google cnn
    structure optimization.
    """
    args = google_topo_api.specified_task(problem)
    if cnn_kwargs is None:
        cnn_kwargs = {}

    model = google_models.PixelModel(args=args)
    ds_mma = google_train.method_of_moving_asymptotes(model, max_iterations)

    model = google_models.CNNModel(args=args, **cnn_kwargs)
    ds_cnn = google_train.train_lbfgs(model, max_iterations)

    dims = pd.Index(["cnn-lbfgs", "mma"], name="model")
    return xarray.concat([ds_cnn, ds_mma], dim=dims)


def tounn_train_and_outputs(problem, requires_flip):
    """
    Function that will run the TOuNN pipeline
    """
    # Try setting seed here as well
    models.set_seed(0)

    # Get the problem name
    problem_name = problem.name

    # Get the arguments for the problem
    args = topo_api.specified_task(problem)

    # Get the problem dimensions
    # Our arguments come as torch tensors so we need to convert back
    # to numpy and int to work with their framework
    nelx, nely = int(args['nelx'].numpy()), int(args['nely'].numpy())

    # Get the volume fraction
    desiredVolumeFraction = args['volfrac']

    # Get the forces - convert to numpy to work with
    # their pipeline
    force = args['forces'].cpu().numpy()

    # Add an extra axis because their framework expects
    # (n, 1)
    force = force[:, None]

    # Get the fixed dofs
    fixed = args['fixdofs'].cpu().numpy()

    # Get epsilon value
    epsilon = args['epsilon']

    # TODO: Figure out how this works with non-design
    # regions but for now it will be none
    nonDesignRegion = {
        'Rect': None,
        'Circ': None,
        'Annular': None,
    }

    # Symmetry about axes
    symXAxis = False
    symYAxis = False

    # Penal in their code starts at 2
    penal = 2

    # Neural network config
    numLayers = 5
    numNeuronsPerLyr = 20
    minEpochs = 20
    maxEpochs = 2000
    useSavedNet = False

    # Run the pipeline
    topOpt = TopologyOptimizer()

    # Initialize the FE (Finite element) solver
    topOpt.initializeFE(problem.name, nelx, nely, force, fixed, penal, nonDesignRegion)

    # Initialize the optimizer
    topOpt.initializeOptimizer(
        numLayers, numNeuronsPerLyr, desiredVolumeFraction, symXAxis, symYAxis
    )

    # Run the optimization
    topOpt.optimizeDesign(maxEpochs, minEpochs, useSavedNet)

    # After everything is fitted we need to extract the final information
    # Set the plotResolution to 1
    plotResolution = 1

    # compute the points for the problem
    xyPlot, nonDesignPlotIdx = topOpt.generatePoints(
        topOpt.FE.nelx, topOpt.FE.nely, plotResolution, topOpt.nonDesignRegion
    )

    # Compute the final density
    density = torch.flatten(topOpt.topNet(xyPlot, nonDesignPlotIdx))
    density = density.detach().cpu().numpy()
    best_final_design = density.copy()

    # The final design needs to be reshaped and transposed
    best_final_design = best_final_design.reshape(
        plotResolution * topOpt.FE.nelx, plotResolution * topOpt.FE.nely
    )
    best_final_design = best_final_design.T

    # get the best score
    best_score = np.round(topOpt.convergenceHistory[-1][-1], 2)

    # Return everything
    mask = (torch.broadcast_to(args["mask"], (nely, nelx)) > 0).cpu().numpy()

    # Compute the binary constraint
    binary_constraint = calculate_binary_constraint(
        design=best_final_design,
        mask=mask,
        epsilon=epsilon,
    )
    volume_constraint = calculate_volume_constraint(
        design=best_final_design,
        mask=mask,
        volume=desiredVolumeFraction,
    )

    # Add more information about the outputs
    if requires_flip:
        if (
            ("mbb" in problem_name)
            or ("l_shape" in problem_name)
            or ("cantilever" in problem_name)
        ):
            best_final_design = np.hstack(
                [best_final_design[:, ::-1], best_final_design]
            )

        if (
            ("multistory" in problem_name)
            or ("thin" in problem_name)
            or ("michell" in problem_name)
        ):
            best_final_design = np.hstack(
                [best_final_design, best_final_design[:, ::-1]] * 2
            )

    return best_final_design, best_score, binary_constraint, volume_constraint


@cli.command('run-multi-structure-pipeline')
def run_multi_structure_pipeline():
    """
    Task that will build out multiple structures and compare
    performance against known benchmarks.
    """
    # Model size
    model_size = 'medium'

    # CNN parameters
    cnn_features = (256, 128, 64, 32, 16)

    # Configurations
    configs = {
        'tiny': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 6 for features in cnn_features),
        },
        'xsmall': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 5 for features in cnn_features),
        },
        'small': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 4 for features in cnn_features),
        },
        'medium': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 3 for features in cnn_features),
        },
        'large': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 2 for features in cnn_features),
        },
        # x-large has been our original architecture
        'xlarge': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 1 for features in cnn_features),
        },
    }

    # CNN kwargs
    cnn_kwargs = configs[model_size]

    # Set seed
    models.set_seed(0)  # Model seed is set here but results are changing?

    # For testing we will run two experimentation trackers
    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Topology-Optimization'

    # Enable wandb
    wandb.login(key=API_KEY)

    # Initalize wandb
    # TODO: Save training and validation curves per fold
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=['topology-optimization-task', model_size],
        config=cnn_kwargs,
    )

    # Get the device to be used
    device = utils.get_devices()
    num_trials = 1
    maxit = 2000
    max_iterations = 200

    # Set up the problem names
    problem_config = [
        # # Medium Size Problems
        ("mbb_beam_96x32_0.5", True, 1, 50),
        ("cantilever_beam_full_96x32_0.4", True, 1, 50),
        ("michell_centered_top_64x128_0.12", True, 1, 50),
        ("l_shape_0.4_128x128_0.3", True, 1, 50),
        # # Large Size Problems
        # ("mbb_beam_384x128_0.3", True, 1, 50),
        # ("cantilever_beam_full_384x128_0.2", True, 1, 50),
        # ("michell_centered_top_128x256_0.12", True, 1, 50),
    ]

    # renaming
    name_mapping = {
        # Medium Size Problems
        'mbb_beam_96x32_0.5': 'MBB Beam \n $96\\times32; v_f = 0.5$',
        'cantilever_beam_full_96x32_0.4': 'Cantilever Beam \n $96\\times32; v_f=0.4$',
        'michell_centered_top_64x128_0.12': 'Michell Top \n $64\\times128; v_f=0.12$',
        'thin_support_bridge_128x128_0.2': 'Thin Support Bridge \n $128\\times128; v_f=0.2$',  # noqa
        'l_shape_0.4_128x128_0.3': 'L-Shape 0.4 \n $128\\times128; v_f=0.3$',
        # Large Size Problems
        'mbb_beam_384x128_0.3': 'MBB Beam \n $384\\times128; v_f = 0.3$',
        'cantilever_beam_full_384x128_0.2': 'Cantilever Beam \n $384\\times128; v_f=0.2$',  # noqa
        'michell_centered_top_128x256_0.12': 'Michell Top \n $128\\times256; v_f=0.12$',
    }

    # PyGranso function
    comb_fn = train.volume_constrained_structural_optimization_function

    # Build the problems for pygranso and google
    PYGRANSO_PROBLEMS_BY_NAME = problems.build_problems_by_name(device=device)

    # For running this we only want one trial
    # with maximum iterations 1000
    structure_outputs = []
    for problem_name, requires_flip, total_frames, cax_size in problem_config:
        print(f"Building structure: {problem_name}")
        problem = PYGRANSO_PROBLEMS_BY_NAME.get(problem_name)

        # Get volume assignment
        args = topo_api.specified_task(problem, device=device)
        volume = args["volfrac"]

        nely = int(args["nely"])
        nelx = int(args["nelx"])
        mask = (torch.broadcast_to(args["mask"], (nely, nelx)) > 0).cpu().numpy()

        # Build the structure with pygranso
        outputs = train.train_pygranso(
            problem=problem,
            device=device,
            pygranso_combined_function=comb_fn,
            requires_flip=requires_flip,
            total_frames=total_frames,
            cnn_kwargs=cnn_kwargs,
            neptune_logging=None,
            num_trials=num_trials,
            maxit=maxit,
        )

        # Build the outputs
        pygranso_outputs = build_outputs(
            problem_name=problem_name,
            outputs=outputs,
            mask=mask,
            volume=volume,
            requires_flip=requires_flip,
        )

        # Add TOuNN to the pipeline
        tounn_outputs = tounn_train_and_outputs(problem, requires_flip)

        # Build google results - lets use our problem library
        # so we can also have custom structures not in the
        # google code
        google_problem = problem

        # Set to numpy for google framework
        google_problem.normals = google_problem.normals.cpu().numpy()
        google_problem.forces = google_problem.forces.cpu().numpy()

        if not isinstance(google_problem.mask, int):
            google_problem.mask = google_problem.mask.cpu().numpy()

        google_problem.mirror_left = True
        google_problem.mirror_right = False

        ds = train_all(google_problem, max_iterations)

        # Get google outputs
        benchmark_outputs = build_google_outputs(
            problem_name=problem_name,
            ds=ds,
            mask=mask,
            volume=volume,
            requires_flip=requires_flip,
        )

        # Zip the results together to create a dataframe
        google_cnn_outputs = benchmark_outputs["google-cnn"]
        mma_outputs = benchmark_outputs["mma"]

        # All outputs
        outputs = pd.DataFrame(
            zip(
                pygranso_outputs,
                tounn_outputs,
                google_cnn_outputs,
                mma_outputs,
            ),
        )
        outputs = outputs.transpose()
        outputs.columns = ["designs", "loss", "binary_constraint", "volume_constraint"]
        outputs["problem_name"] = problem_name

        # Add titles
        titles = ["PG", "TOuNN", "Google", "MMA"]
        outputs["titles"] = titles
        outputs["cax_size"] = cax_size
        structure_outputs.append(outputs)

        gc.collect()
        torch.cuda.empty_cache()

    print('Building and saving outputs, hang tight! ⏳')
    # Concat all structures
    structure_outputs = pd.concat(structure_outputs)
    structure_outputs["loss"] = structure_outputs["loss"].astype(float)

    # Create the output plots
    fig, axes = plt.subplots(
        4, len(problem_config), figsize=(13, 6), constrained_layout=True
    )
    axes = axes.T.flatten()
    fig.subplots_adjust(wspace=0, hspace=0)

    # add the axes to the dataframe
    structure_outputs["ax"] = axes

    # Create the color map
    # color_map = {
    #     0: ('yellow', 'black'),  # Best
    #     1: ('orange', 'black'),
    #     2: ('darkviolet', 'white'),
    #     3: ('navy', 'white'),  # Worst
    # }

    # Minnesota color map
    color_map = {
        0: ('gold', 'black'),  # Best
        1: ('orange', 'black'),
        2: ('maroon', 'white'),
        3: ('silver', 'black'),  # Worst
    }

    # Get the best to worst
    structure_outputs["initial_order"] = structure_outputs.groupby(
        "problem_name"
    ).cumcount()
    structure_outputs = structure_outputs.sort_values(
        ["problem_name", "loss"]
    ).reset_index(drop=True)
    structure_outputs["order"] = structure_outputs.groupby("problem_name").cumcount()
    structure_outputs = structure_outputs.sort_values(
        ["problem_name", "initial_order"]
    )  # noqa
    structure_outputs["formatting"] = structure_outputs["order"].map(color_map)

    # Will create directories for saving models
    save_path = os.path.join(
        '/home/jusun/dever120/NCVX-Neural-Structural-Optimization/results',
        f'{wandb.run.id}',
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"The directory {save_path} was created.")
    else:
        print(f"The directory {save_path} already exists.")

    # Save the data
    structure_outputs[["problem_name", "loss", "initial_order", "formatting"]].to_csv(
        os.path.join(save_path, 'structure_outputs.csv'), index=False
    )

    # Sort the structures and algorithms
    structure_outputs = structure_outputs.sort_values(
        ['problem_name', 'titles']
    ).reset_index(drop=True)
    structure_outputs['problem_name'] = structure_outputs['problem_name'].map(
        name_mapping
    )

    for index, data in enumerate(structure_outputs.itertuples()):
        ax = axes[index]
        ax.imshow(data.designs, cmap='Greys', aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])

        # Set y-labels
        if index in [0, 1, 2, 3]:
            ax.set_ylabel(data.titles, fontsize=9, weight='bold')

        # Set x-labels
        if index in [0, 4, 8, 12, 16]:
            ax.set_title(data.problem_name, weight='bold', fontsize=9)

        # Add the colors box for the scoring
        divider = make_axes_locatable(ax)

        cax = divider.append_axes("bottom", size="40%", pad=0.01)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)

        formatting = data.formatting
        facecolor = formatting[0]
        fontcolor = formatting[1]

        # Set the face color of the box
        cax.set_facecolor(facecolor)
        cax.spines["bottom"].set_color(facecolor)
        cax.spines["top"].set_color(facecolor)
        cax.spines["right"].set_color(facecolor)
        cax.spines["left"].set_color(facecolor)

        text = f"{data.loss} / {data.binary_constraint} / {data.volume_constraint}"
        cax.text(
            0.5,
            0.5,
            text,
            ha='center',
            va='center',
            fontsize=8,
            color=fontcolor,
            weight='bold',
        )

    # Save the fig
    fig.savefig(
        f'/home/jusun/dever120/NCVX-Neural-Structural-Optimization/results/{model_size}-results.png',  # noqa
        bbox_inches='tight',
    )

    # Save figure to weights and biases
    wandb.log({'plot': wandb.Image(fig)})

    print('Run completed! 🎉')


@cli.command('run-multi-structure-pygranso-pipeline')
def run_multi_structure_pygranso_pipeline():
    """
    Task that will build out multiple structures and compare
    performance against known benchmarks.
    """
    # Model size
    model_size = 'medium'

    # CNN parameters
    cnn_features = (256, 128, 64, 32, 16)

    # Configurations
    configs = {
        'tiny': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 6 for features in cnn_features),
        },
        'xsmall': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 5 for features in cnn_features),
        },
        'small': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 4 for features in cnn_features),
        },
        'medium': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 3 for features in cnn_features),
        },
        'large': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 2 for features in cnn_features),
        },
        # x-large has been our original architecture
        'xlarge': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 1 for features in cnn_features),
        },
    }

    # CNN kwargs
    cnn_kwargs = configs[model_size]

    # Set seed
    models.set_seed(0)  # Model seed is set here but results are changing?

    # For testing we will run two experimentation trackers
    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Topology-Optimization'

    # Enable wandb
    wandb.login(key=API_KEY)

    # Initalize wandb
    # TODO: Save training and validation curves per fold
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=['topology-optimization-task', model_size],
        config=cnn_kwargs,
    )

    # Get the device to be used
    device = utils.get_devices()
    num_trials = 4
    maxit = 1

    # Set up the problem names
    problem_config = [
        # # Medium Size Problems
        ("mbb_beam_96x32_0.5", True, 1, 50),
        ("cantilever_beam_full_96x32_0.4", True, 1, 50),
        ("michell_centered_top_64x128_0.12", True, 1, 50),
        ("l_shape_0.4_128x128_0.3", True, 1, 50),
    ]

    # renaming
    name_mapping = {
        # Medium Size Problems
        'mbb_beam_96x32_0.5': 'MBB Beam \n $96\\times32; v_f = 0.5$',
        'cantilever_beam_full_96x32_0.4': 'Cantilever Beam \n $96\\times32; v_f=0.4$',
        'michell_centered_top_64x128_0.12': 'Michell Top \n $64\\times128; v_f=0.12$',
        'thin_support_bridge_128x128_0.2': 'Thin Support Bridge \n $128\\times128; v_f=0.2$',  # noqa
        'l_shape_0.4_128x128_0.3': 'L-Shape 0.4 \n $128\\times128; v_f=0.3$',
    }

    # PyGranso function
    comb_fn = train.volume_constrained_structural_optimization_function

    # Build the problems for pygranso and google
    PYGRANSO_PROBLEMS_BY_NAME = problems.build_problems_by_name(device=device)

    # For running this we only want one trial
    # with maximum iterations 1000
    structure_outputs = []
    for problem_name, requires_flip, total_frames, cax_size in problem_config:
        print(f"Building structure: {problem_name}")
        problem = PYGRANSO_PROBLEMS_BY_NAME.get(problem_name)

        # Build the structure with pygranso
        outputs = train.train_pygranso(
            problem=problem,
            device=device,
            pygranso_combined_function=comb_fn,
            requires_flip=requires_flip,
            total_frames=total_frames,
            cnn_kwargs=cnn_kwargs,
            neptune_logging=None,
            num_trials=num_trials,
            maxit=maxit,
        )
        structure_outputs.append(outputs)

    # Plot the different seeds
    fig, axes = plt.subplots(len(problem_config), num_trials, figsize=(15, 8))
    fig.subplots_adjust(wspace=0, hspace=0)

    # Get the updated structure names
    structures = list(name_mapping.values())

    # iterate over the structures and plot different seeds
    for index, structure_name in enumerate(structures):
        # Get the designs
        designs = structure_outputs[index]['designs']

        # Get the losses
        losses = structure_outputs[index]['losses']
        losses = pd.DataFrame(losses).ffill()
        losses = losses.iloc[-1, :]
        print(f'All losses {losses}')
        print('\n')
        print(f'Avg loss for {structure_name}; {losses.mean()}')

        for trial in range(num_trials):
            ax = axes[index, trial]
            design = designs[index, :, :]
            design = np.hstack((design[:, ::-1], design))
            ax.imshow(design, cmap='Greys', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])

            # Add the ylabel (structure name)
            if trial == 0:
                ax.set_ylabel(f'{structure_name}', fontsize=14, weight='bold')

    # Also, save fig
    fig.savefig(
        f'/home/jusun/dever120/NCVX-Neural-Structural-Optimization/results/{model_size}-pygranso-seed-results.png',  # noqa
        bbox_inches='tight',
    )


if __name__ == "__main__":
    cli()
