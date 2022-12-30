#!/usr/bin/python
# third party
import click
import matplotlib.pyplot as plt
import neptune.new as neptune
import neural_structural_optimization.problems as google_problems
import numpy as np
import pandas as pd
import torch

# first party
import problems
import train
import utils


# Run the tasks
@click.command()
@click.option("--problem_name", default="mbb_beam", type=click.STRING)
@click.option("--num_trials", default=50)
@click.option("--maxit", default=1500)
@click.option("--resizes", is_flag=True, default=False)
def structural_optimization_task(problem_name, num_trials, maxit, resizes):
    click.echo(problem_name)
    # Enable the neptune run
    # TODO: make the api token an environment variable
    run = neptune.init_run(
        project="dever120/CNN-Structural-Optimization-Pytorch",
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
    GOOGLE_PROBLEMS_BY_NAME = google_problems.PROBLEMS_BY_NAME

    # Setup the problem
    problem = PROBLEMS_BY_NAME.get(problem_name)
    google_problem = GOOGLE_PROBLEMS_BY_NAME.get(problem_name)

    if (problem.name is None) or (google_problem.name is None):
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
    }

    # Run the trials
    outputs = train.train_pygranso(
        problem=problem,
        device=device,
        pygranso_combined_function=comb_fn,
        cnn_kwargs=cnn_kwargs,
        neptune_logging=run,
        num_trials=num_trials,
        maxit=maxit,
    )

    # Set up the loss dataframe
    losses_df = pd.DataFrame(outputs["losses"]).ffill()

    # Get all of the final losses
    final_losses = losses_df.iloc[-1, :].values

    # Argsort losses - largest to smallest
    losses_indexes = np.argsort(final_losses)[::-1]
    final_losses = final_losses[losses_indexes]
    final_designs = outputs["designs"][losses_indexes, :, :]
    initial_volumes = outputs["trials_initial_volumes"][losses_indexes]

    # Save the best final design
    best_final_design = final_designs[-1, :, :]
    best_score = np.round(final_losses[-1], 2)

    # TODO: Will need a special implmentation for some of the final
    # designs
    fig = utils.build_final_design(
        problem.name, best_final_design, best_score, figsize=(10, 6)
    )
    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    run[f"best_trial-{problem.name}-final-design"].upload(fig)

    # Close the figure
    plt.close()

    # Create a histogram of losses and scatter plot of volumes
    meta_df = pd.DataFrame(
        {
            "final_losses": final_losses,
            "initial_volumes": initial_volumes,
        }
    )

    # plot new information
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))

    # plot the initial volumes vs the final losses
    axes = axes.flatten()
    ax1 = axes[0]
    meta_df.plot.scatter(x="final_losses", y="initial_volumes", ax=ax1)
    ax1.axhline(problem.density, color="r")
    ax1.set_title("Final Compliance vs. Initial Volumes")
    ax1.set_xlabel("Compliance")
    ax1.set_ylabel("Initial Volume")
    ax1.grid()

    # Plot the histogram of the final losses
    ax2 = axes[1]
    meta_df["final_losses"].hist(density=True, ax=ax2)
    ax2.set_title("Histogram of Final Compliances")
    ax2.set_xlabel("Compliance")
    ax2.set_ylabel("Probability")

    # Plot the median
    losses_median = meta_df["final_losses"].median()
    losses_mean = meta_df["final_losses"].mean()
    ax2.axvline(losses_median, color="green", label="median")
    ax2.axvline(losses_mean, color="orange", label="mean")
    ax2.legend()
    ax2.grid()

    fig.tight_layout()

    # push to neptune
    run[f"statistics-and-comparisons"].upload(fig)
    run["median-compliance"].log(losses_median)
    run["mean-compliance"].log(losses_mean)
    plt.close()

    # # Train the google problem
    # print(f"Training Google - {google_problem.name}")
    # google_trials = train.train_google(
    #     google_problem,
    #     maxit,
    #     cnn_kwargs=cnn_kwargs,
    #     num_trials=num_trials,
    #     neptune_logging=run,
    # )
    # print("Finished training")

    # # Google best trial
    # google_best_trial = sorted(google_trials, key=lambda x: x[0])[0]

    # # Get the losses
    # google_best_score = np.round(google_best_trial[0], 2)

    # # Next extract the final image
    # google_design = google_best_trial[2]
    # google_design = np.squeeze(google_design)

    # # Plot the google image
    # google_fig = utils.build_final_design(
    #     google_problem.name, google_design, google_best_score, figsize=(10, 6)
    # )
    # google_fig.subplots_adjust(hspace=0)
    # google_fig.tight_layout()
    # run[f"google-best_trial-{problem.name}-final-design"].upload(google_fig)

    # plt.close()

    # # Get the losses for pygranso and google
    # pygranso_losses = [losses for _, losses, _, _ in trials]
    # google_losses = [losses for _, losses, _, _ in google_trials]

    # trials_dict = {"pygranso_losses": pygranso_losses, "google_losses": google_losses}

    # # Build and save the losses data for this run
    # utils.build_loss_plots(problem.name, trials_dict, run)

    run.stop()


if __name__ == "__main__":
    structural_optimization_task()
