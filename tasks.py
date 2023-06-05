#!/usr/bin/python

# third party
import click
import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import pandas as pd
import torch
import xarray
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
from neural_structural_optimization import models as google_models
from neural_structural_optimization import pipeline_utils
from neural_structural_optimization import problems as google_problems
from neural_structural_optimization import topo_api as google_topo_api
from neural_structural_optimization import train as google_train

# first party
import problems
import topo_api
import train
import utils


def calculate_binary_constraint(design, epsilon):
    """
    Function to compute the binary constraint
    """
    return np.round(np.mean(design * (1 - design)) - epsilon, 4)


def calculate_volume_constraint(design, volume):
    """
    Function that computes the volume constraint
    """
    return np.round(np.mean(design) / volume - 1.0, 4)


def build_outputs(problem_name, outputs, volume, requires_flip, epsilon=1e-3):
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
    if requires_flip:
        if ("mbb" in problem_name) or ("thin" in problem_name):
            best_final_design = np.hstack(
                [best_final_design[:, ::-1], best_final_design]
            )

        elif "multistory" in problem_name:
            best_final_design = np.hstack(
                [best_final_design, best_final_design[:, ::-1]] * 2
            )

    # last row, first column (-1, 0)
    best_score = np.round(losses_df.iloc[-1, 0], 2)

    # Compute the binary and volume constraints
    binary_constraint = calculate_binary_constraint(
        design=best_final_design,
        epsilon=epsilon,
    )

    # volume constraint
    volume_constraint = calculate_volume_constraint(
        design=best_final_design,
        volume=volume,
    )

    return best_final_design, best_score, binary_constraint, volume_constraint


def build_google_outputs(problem_name, ds, volume, requires_flip, epsilon=1e-3):
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
        design=cnn_final_design, epsilon=epsilon
    )
    cnn_volume_constraint = calculate_volume_constraint(
        design=cnn_final_design,
        volume=volume,
    )

    # MMA final design
    mma_final_design = final_designs[1, :, :]
    mma_binary_constraint = calculate_binary_constraint(
        design=mma_final_design,
        epsilon=epsilon,
    )
    mma_volume_constraint = calculate_volume_constraint(
        design=mma_final_design,
        volume=volume,
    )

    if requires_flip:
        if ("mbb" in problem_name) or ("thin" in problem_name):
            cnn_final_design = np.hstack([cnn_final_design[:, ::-1], cnn_final_design])
            mma_final_design = np.hstack([mma_final_design[:, ::-1], mma_final_design])

        if "multistory" in problem_name:
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
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYmIwMTUyNC05YmZmLTQ1NzctOTEyNS1kZTIxYjU5NjY5YjAifQ==",
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
    initial_volumes = outputs["trials_initial_volumes"][losses_indexes]

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
    ax2.set_ylabel("Binary Constraint: $x \in [0, 1]$")

    # Set xlabel
    ax1.set_xlabel("Iteration")
    ax1.set_title("Compliance & Binary Constraint @ t")
    ax1.grid()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    run[f"best-trial-{problem.name}-compliance-&-binary-constraint"].upload(fig)

    # Close the figure
    plt.close()

    # Create a histogram of losses and scatter plot of volumes
    meta_df = pd.DataFrame(
        {
            "final_losses": losses,
            "initial_volumes": initial_volumes,
        }
    )

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


def run_multi_structure_pipeline():
    """
    Task that will build out multiple structures and compare
    performance against known benchmarks.
    """
    # Get the device to be used
    device = utils.get_devices()
    num_trials = 1
    maxit = 500

    # Set up the problem names
    problem_config = [
        ("mbb_beam_96x32_0.5", True, 1, 66),
        ("multistory_building_64x128_0.4", True, 2, 30),
        ("thin_support_bridge_128x128_0.2", True, 1, 30),
        ("l shape_0.2_128x128_0.3", False, 1, 30),
        ("l_shape_0.4_128x128_0.3", False, 1, 30),
    ]

    # PyGranso function
    comb_fn = train.volume_constrained_structural_optimization_function

    # Build the problems for pygranso and google
    PYGRANSO_PROBLEMS_BY_NAME = problems.build_problems_by_name(device=device)

    # For running this we only want one trial
    # with maximum iterations 1000
    structure_outputs = []
    for (problem_name, requires_flip, total_frames, cax_size) in problem_config:
        print(f"Building structure: {problem_name}")
        pygranso_problem = PYGRANSO_PROBLEMS_BY_NAME.get(problem_name)

        # Get volume assignment
        args = topo_api.specified_task(pygranso_problem, device=device)
        volume = args["volfrac"]

        # Build the structure with pygranso
        outputs = train.train_pygranso(
            problem=pygranso_problem,
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
            volume=volume,
            requires_flip=requires_flip,
        )

        # Build google results
        google_problem = google_problems.PROBLEMS_BY_NAME[problem_name]
        max_iterations = 200
        ds = train_all(google_problem, max_iterations)

        # Get google outputs
        benchmark_outputs = build_google_outputs(
            problem_name=problem_name, ds=ds, volume=volume, requires_flip=requires_flip
        )

        # Zip the results together to create a dataframe
        google_cnn_outputs = benchmark_outputs["google-cnn"]
        mma_outputs = benchmark_outputs["mma"]

        # All outputs
        outputs = pd.DataFrame(
            zip(
                pygranso_outputs,
                google_cnn_outputs,
                mma_outputs,
            ),
        )
        outputs = outputs.transpose()
        outputs.columns = ["designs", "loss", "binary_constraint", "volume_constraint"]
        outputs["problem_name"] = problem_name

        # Create the color map
        color_map = {
            0: ("yellow", "black"),
            1: ("orange", "black"),
            2: ("mediumblue", "white"),
        }

        # Get the best to worst
        outputs["formatting"] = outputs.groupby("problem_name", group_keys=False)[
            "loss"
        ].apply(lambda x: np.argsort(x))
        outputs["formatting"] = outputs["formatting"].map(color_map)

        # Add titles
        titles = ["PyGranso-CNN", f"{problem_name} \n Google-CNN", "MMA"]
        outputs["titles"] = titles
        outputs["cax_size"] = cax_size
        structure_outputs.append(outputs)

    # Concat all structures
    structure_outputs = pd.concat(structure_outputs)

    # Create the output plots
    fig, axes = plt.subplots(len(problem_config), 3, figsize=(10, 8))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.1, wspace=0.01)

    # add the axes to the dataframe
    structure_outputs["ax"] = axes

    for index, data in enumerate(structure_outputs.itertuples()):
        ax = data.ax
        ax.imshow(data.designs, cmap="Greys")

        # Add the colors box for the scoring
        divider = make_axes_locatable(ax)

        cax = divider.append_axes("bottom", size=f"{data.cax_size}%", pad=0.01)
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
        at = AnchoredText(
            text,
            loc=10,
            frameon=False,
            prop=dict(
                backgroundcolor=facecolor,
                size=10,
                color=fontcolor,
            ),
        )
        cax.add_artist(at)
        ax.set_axis_off()
        ax.set_title(data.titles, fontsize=14)


if __name__ == "__main__":
    structural_optimization_task()
