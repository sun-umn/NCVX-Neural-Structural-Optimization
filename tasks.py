#!/usr/bin/python
# third party
import click
import neptune.new as neptune
import numpy as np

# first party
import problems
import train
import utils


# Run the tasks
@click.command()
@click.option("--problem_name", default="mbb_beam", type=click.STRING)
@click.option("--height", default=20)
@click.option("--width", default=60)
@click.option("--interval", default=16)
@click.option("--density", default=0.5)
@click.option("--alpha", default=5e3)
@click.option("--num_trials", default=50)
@click.option("--maxit", default=1500)
def structural_optimization_task(
    problem_name, height, width, interval, density, alpha, num_trials, maxit
):
    click.echo(problem_name)
    # Enable the neptune run
    # TODO: make the api token an environment variable
    run = neptune.init_run(
        project="dever120/CNN-Structural-Optimization-Pytorch",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYmIwMTUyNC05YmZmLTQ1NzctOTEyNS1kZTIxYjU5NjY5YjAifQ==",
    )

    # Get the available device
    device = utils.get_devices()

    # PyGranso Volume Function
    comb_fn = train.volume_constrained_structural_optimization_function

    # Initialize the problem to be solved
    if problem_name == "mbb_beam":
        problem = problems.mbb_beam(
            height=height, width=width, density=density, device=device
        )
        problem.name = f"mbb_beam_{width}x{height}_{density}"

        # CNN Kwargs
        cnn_kwargs = dict(resizes=(1, 1, 2, 2, 1))
    elif problem_name == "multistory_building":
        problem = problems.multistory_building(
            width=width,
            height=height,
            density=density,
            interval=interval,
            device=device,
        )
        problem.name = f"multistory_building_{width}x{height}_{density}"

        # CNN Kwargs
        cnn_kwargs = None
    elif problem_name == "thin_support_bridge":
        problem = problems.thin_support_bridge(
            width=width,
            height=height,
            density=density,
            device=device,
        )
        problem.name = f"thin_support_bridge_{width}x{height}_{density}"

        # CNN Kwargs
        cnn_kwargs = None
    else:
        raise ValueError("Structure is not valid!")

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
        "alpha": alpha,
    }

    # Run the trials
    trials = train.train_pygranso(
        problem=problem,
        device=device,
        pygranso_combined_function=comb_fn,
        cnn_kwargs=cnn_kwargs,
        neptune_logging=run,
        num_trials=num_trials,
        maxit=maxit,
        alpha=alpha,
    )

    # Define the best trial
    best_trial = sorted(trials)[0]

    # Build and save the losses data for this run
    utils.build_trial_loss_plot(problem.name, trials, run)

    # Save the best final design
    best_final_design = best_trial[2]
    best_score = np.round(best_trial[0], 2)

    # TODO: Will need a special implmentation for some of the final
    # designs
    fig = utils.build_final_design(
        problem.name, best_final_design, best_score, figsize=(10, 6)
    )
    run[f"best_trial-{problem.name}-final-design"].upload(fig)

    run.stop()


if __name__ == "__main__":
    structural_optimization_task()
