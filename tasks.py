#!/usr/bin/python
# stdlib
import gc
import os
import warnings

# third party
import click
import matplotlib.pyplot as plt
import neptune.new as neptune
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
        if (
            ("mbb" in problem_name)
            or ("l_shape" in problem_name)
            or ("cantilever" in problem_name)
        ):
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


# Run the tasks
@click.command()
@click.option("--problem_name", default="mbb_beam", type=click.STRING)
@click.option("--num_trials", default=50)
@click.option("--maxit", default=1500)
@click.option("--requires_flip", is_flag=True, default=False)
@click.option("--total_frames", default=1)
@click.option("--resizes", is_flag=True, default=False)
@click.option("--note", default="original", type=click.STRING)
def structural_optimization_task(
    problem_name, num_trials, maxit, requires_flip, total_frames, resizes, note
):
    click.echo(problem_name)
    # Enable the neptune run
    # TODO: make the api token an environment variable
    run = neptune.init_run(
        project="dever120/CNN-Structural-Optimization-Prod",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYmIwMTUyNC05YmZmLTQ1NzctOTEyNS1kZTIxYjU5NjY5YjAifQ==",  # noqa
    )

    # Get the available device
    device = utils.get_devices()
    # device = torch.device('cpu')

    # PyGranso Volume Function
    comb_fn = train.volume_constrained_structural_optimization_function

    # Consider resizes
    if resizes:
        cnn_kwargs = dict(resizes=(1, 1, 2, 2, 1))
    else:
        cnn_kwargs = None
    print(f"Resizes = {cnn_kwargs}")

    # Build the problems by name with the correct device
    PROBLEMS_BY_NAME = problems.build_problems_by_name(device=device)
    # GOOGLE_PROBLEMS_BY_NAME = google_problems.PROBLEMS_BY_NAME

    # Setup the problem
    problem = PROBLEMS_BY_NAME.get(problem_name)
    # google_problem = GOOGLE_PROBLEMS_BY_NAME.get(problem_name)

    if problem.name is None:  # or (google_problem.name is None):
        raise ValueError(f"{problem_name} is not an elgible structure")

    # Add a tag for each type of problem as well
    run["sys/tags"].add([problem.name])

    # num trials
    num_trials = num_trials

    # max iterations
    maxit = maxit

    # Save the parameters
    run["parameters"] = {
        "problem_name": problem.name,
        "num_trials": num_trials,
        "maxit": maxit,
        "cnn_kwargs": cnn_kwargs,
        "device": device,
        "note": note,
    }

    # Run the trials
    outputs = train.train_pygranso(
        problem=problem,
        device=device,
        pygranso_combined_function=comb_fn,
        requires_flip=requires_flip,
        total_frames=total_frames,
        cnn_kwargs=cnn_kwargs,
        neptune_logging=run,
        num_trials=num_trials,
        maxit=maxit,
    )

    # Set up the loss dataframe
    losses_df = pd.DataFrame(outputs["losses"])
    volumes_df = pd.DataFrame(outputs["volumes"])
    binary_constraint_df = pd.DataFrame(outputs["binary_constraint"])

    # Get all of the final losses
    losses = np.min(losses_df.ffill(), axis=0).values

    # Argsort losses - smallest to largest
    losses_indexes = np.argsort(losses)
    losses_df = losses_df.iloc[:, losses_indexes]
    volumes_df = volumes_df.iloc[:, losses_indexes]
    binary_constraint_df = binary_constraint_df.iloc[:, losses_indexes]
    final_designs = outputs["designs"][losses_indexes, :, :]

    # Save the best final design
    best_final_design = final_designs[0, :, :]
    best_score = np.round(losses_df.ffill().iloc[-1, 0], 2)

    # TODO: Will need a special implmentation for some of the final
    # designs
    fig = utils.build_final_design(
        problem.name,
        best_final_design,
        best_score,
        requires_flip,
        total_frames=total_frames,
        figsize=(10, 6),
    )
    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    run[f"best_trial-{problem.name}-final-design"].upload(fig)

    # Write a histogram as well
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    hist_values = pd.Series(best_final_design.flatten())
    hist_values.hist(bins=50, density=True, color="blue", ax=ax)
    ax.set_title("$x$ Material Distribution (Binary Constraint)")
    run[f"best-trial-{problem.name}-final-design-histogram"].upload(fig)
    plt.close()

    # Create a figure with the volumes and compliance values from
    # the best seed
    best_trial_losses = losses_df.iloc[:, 0]
    best_trial_losses.name = "compliance"

    best_trial_volume_constr = volumes_df.iloc[:, 0]
    best_trial_volume_constr.name = "volume"

    # Let's build the plot
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    # Set up the second axis
    ax2 = ax1.twinx()

    # Plot the data
    # Plot the compliance
    best_trial_losses.plot(color="red", lw=2, marker="*", ax=ax1, label="compliance")
    ax1.set_ylabel("Compliance")

    # Plot the volume constraint
    best_trial_volume_constr.plot(
        color="blue", lw=2, marker="o", ax=ax2, label="volume"
    )
    ax2.set_ylabel("Volume / $V_t$")

    # Set xlabel
    ax1.set_xlabel("Iteration")
    ax1.set_title("Compliance & Volume @ t")
    ax1.grid()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    run[f"best-trial-{problem.name}-compliance-&-volume"].upload(fig)

    # Close the figure
    plt.close()

    # Create a plot for binary constraints also
    best_trial_bc_constr = binary_constraint_df.iloc[:, 0]
    best_trial_bc_constr.name = "binary constraint"

    # Let's build the plot
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    # Set up the second axis
    ax2 = ax1.twinx()

    # Plot the data
    # Plot the compliance
    best_trial_losses.plot(color="red", lw=2, marker="*", ax=ax1, label="compliance")
    ax1.set_ylabel("Compliance")

    # Plot the volume constraint
    best_trial_bc_constr.plot(
        color="blue", lw=2, marker="o", ax=ax2, label="binary constraint"
    )
    ax2.set_ylabel("Binary Constraint: $x \in [0, 1]$")  # noqa

    # Set xlabel
    ax1.set_xlabel("Iteration")
    ax1.set_title("Compliance & Binary Constraint @ t")
    ax1.grid()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    run[f"best-trial-{problem.name}-compliance-&-binary-constraint"].upload(fig)

    # Close the figure
    plt.close()

    run.stop()


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
    maxEpochs = 500
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


def run_multi_structure_pipeline():
    """
    Task that will build out multiple structures and compare
    performance against known benchmarks.
    """
    models.set_seed(0)

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
        tags=['topology-optimization-task'],
    )

    # Get the device to be used
    device = utils.get_devices()
    num_trials = 1
    maxit = 1000
    max_iterations = 200

    # Set up the problem names
    problem_config = [
        ("mbb_beam_96x32_0.5", True, 1, 50),
        ("cantilever_beam_full_96x32_0.4", True, 1, 50),
        ("michell_centered_top_64x128_0.12", True, 1, 50),
        ("thin_support_bridge_128x128_0.2", True, 1, 50),
        ("l_shape_0.2_128x128_0.3", True, 1, 50),
    ]

    # renaming
    name_mapping = {
        'mbb_beam_96x32_0.5': 'MBB Beam \n 96x32; $v_f = 0.5$',
        'cantilever_beam_full_96x32_0.4': 'Cantilever Beam \n 96x32; $v_f=0.4$',
        'michell_centered_top_64x128_0.12': 'Michell Top \n 64x128; $v_f=0.12$',
        'thin_support_bridge_128x128_0.2': 'Thin Support Bridge \n 128x128 \n $v_f=0.2$',  # noqa
        'l_shape_0.2_128x128_0.3': 'L-Shape 0.2 \n 128x128; $v_f=0.3$',
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
            cnn_kwargs=None,
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
        4, len(problem_config), figsize=(12, 4), constrained_layout=True
    )
    axes = axes.T.flatten()
    fig.subplots_adjust(wspace=0)

    # add the axes to the dataframe
    structure_outputs["ax"] = axes

    # Create the color map
    color_map = {
        0: ('yellow', 'black'),
        1: ('orange', 'black'),
        2: ('darkviolet', 'white'),
        3: ('navy', 'white'),
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
    structure_outputs['problem_name'] = structure_outputs.map(name_mapping)

    for index, data in enumerate(structure_outputs.itertuples()):
        ax = axes[index]
        ax.imshow(data.designs, cmap='Greys', aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])

        # Set y-labels
        if index in [0, 1, 2, 3]:
            ax.set_ylabel(data.titles, weight='bold')

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
            fontsize=11,
            color=fontcolor,
            weight='bold',
        )

    # Save figure to weights and biases
    wandb.log({'plot': wandb.Image(fig)})

    # Save to weights and biases
    wandb.log({'plot': wandb.Image(fig)})
    print('Run completed! 🎉')


if __name__ == "__main__":
    # structural_optimization_task()
    run_multi_structure_pipeline()
