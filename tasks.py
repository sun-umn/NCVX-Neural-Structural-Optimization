import click
import neptune.new as neptune

import problems
import train
import utils


# Run the tasks
@click.command()
@click.option("--height", default=20)
@click.option("--width", default=60)
@click.option("--density", default=0.5)
@click.option("--alpha", default=5e3)
@click.option("--num_trials", default=50)
@click.option("--maxit", default=1500)
def mbb_structural_optimization_task(height, width, density, alpha, num_trials, maxit):
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
    problem = problems.mbb_beam(
        height=height, width=width, density=density, device=device
    )
    problem.name = f"mbb_beam_{width}x{height}_{density}"

    # Add a tag for each type of problem as well
    run["sys/tags"].add([problem.name])

    # CNN Kwargs
    cnn_kwargs = dict(resizes=(1, 1, 2, 2, 1))

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

    run.stop()


if __name__ == "__main__":
    mbb_structural_optimization_task()
