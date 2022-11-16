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

    # Define the best trial
    best_trial = sorted(trials)[0]

    # Build and save the losses data for this run
    utils.build_trial_loss_plot(problem.name, trials)

    # Save the final structure from the final design
    utils.build_structure_design(problem.name, best_trial, display="vertical")

    # Save the final design
    utils.build_final_design(problem.name, best_trial)


if __name__ == "__main__":
    structural_optimization_task()
