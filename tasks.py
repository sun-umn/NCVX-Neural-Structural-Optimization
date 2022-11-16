import problems
import train
import utils


# Run the tasks
def structural_optimization_task():
    # Get the available device
    device = utils.get_devices()

    # PyGranso Volume Function
    comb_fn = train.volume_constrained_structural_optimization_function

    # Initialize the problem to be solved
    problem = problems.mbb_beam(height=20, width=60, device=device)
    problem.name = "mbb_beam"

    # CNN Kwargs
    cnn_kwargs = dict(resizes=(1, 1, 2, 2, 1))

    # Run the trials
    trials = train.train_pygranso(
        problem=problem,
        device=device,
        pygranso_combined_function=comb_fn,
        cnn_kwargs=cnn_kwargs,
        num_trials=1,
        maxit=90,
    )


if __name__ == "__main__":
    structural_optimization_task()
