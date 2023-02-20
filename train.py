# stdlib
import gc
import time

# third party
import matplotlib.pyplot as plt
import neural_structural_optimization.models as google_models
import neural_structural_optimization.topo_api as google_api
import neural_structural_optimization.train as google_train
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import xarray
from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

# first party
import models
import topo_api
import topo_physics
import utils


# Updated Section for training PyGranso with Direct Volume constraints
# TODO: We will want to create a more generalized format for training our PyGranso
# problems for for now we will have a separate training process for the volume
# constraint that we have been working on
# Volume constrained function
def volume_constrained_structural_optimization_function(
    model, initial_compliance, ke, args, device, dtype
):
    """
    Combined function for PyGranso for the structural optimization
    problem. The inputs will be the model that reparameterizes x as a function
    of a neural network. V0 is the initial volume, K is the global stiffness
    matrix and F is the forces that are applied in the problem.

    Notes:
    For the original MBB Beam the best alpha is 5e3
    """
    # Initialize the model
    # In my version of the model it follows the similar behavior of the
    # tensorflow repository and only needs None to initialize and output
    # a first value of x

    unscaled_compliance, x_phys, mask = topo_physics.calculate_compliance(
        model, ke, args, device, dtype
    )
    f = 1.0 / initial_compliance * unscaled_compliance

    # Run this problem with no inequality constraints
    ci = None

    ce = pygransoStruct()
    ce.c1 = torch.abs((torch.mean(x_phys[mask]) / args["volfrac"]) - 1.0)  # noqa

    # # Let's try and clear as much stuff as we can to preserve memory
    del x_phys, mask, ke
    gc.collect()
    torch.cuda.empty_cache()

    return f, ci, ce


# Volume constrained function
def multi_material_volume_constrained_structural_optimization_function(
    model, initial_compliance, ke, args, device, dtype
):
    """
    Combined function for PyGranso for the structural optimization
    problem. The inputs will be the model that reparameterizes x as a function
    of a neural network. V0 is the initial volume, K is the global stiffness
    matrix and F is the forces that are applied in the problem.

    Notes:
    For the original MBB Beam the best alpha is 5e3
    """
    # Initialize the model
    # In my version of the model it follows the similar behavior of the
    # tensorflow repository and only needs None to initialize and output
    # a first value of x

    unscaled_compliance, x_phys = topo_physics.calculate_multi_material_compliance(
        model, ke, args, device, dtype
    )
    f = 1.0 / initial_compliance * unscaled_compliance

    # Run this problem with no inequality constraints
    ci = None

    ce = pygransoStruct()
    total_volume = args["nelx"] * args["nely"] * np.max(args["volfrac"])
    ce.c1 = torch.abs((torch.sum(x_phys) / total_volume) - 1.0)  # noqa

    # # Let's try and clear as much stuff as we can to preserve memory
    del x_phys, ke
    gc.collect()
    torch.cuda.empty_cache()

    return f, ci, ce


def train_pygranso(
    problem,
    pygranso_combined_function,
    device,
    requires_flip,
    total_frames,
    cnn_kwargs=None,
    neptune_logging=None,
    *,
    num_trials=50,
    mu=1.0,
    maxit=500,
) -> tuple:
    """
    Function to train structural optimization pygranso
    """
    # Set up the dtypes
    dtype32 = torch.float32
    default_dtype = utils.DEFAULT_DTYPE

    # Get the problem args
    args = topo_api.specified_task(problem, device=device)

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"],
        poisson=args["poisson"],
        device=device,
    )

    # Trials
    trials_designs = np.zeros((num_trials, args["nely"], args["nelx"]))
    trials_losses = np.full((maxit + 1, num_trials), np.nan)
    trials_initial_volumes = []

    for index, seed in enumerate(range(0, num_trials)):
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize the CNN Model
        if cnn_kwargs is not None:
            cnn_model = models.CNNModel(args, **cnn_kwargs).to(
                device=device, dtype=dtype32
            )
        else:
            cnn_model = models.CNNModel(args).to(device=device, dtype=dtype32)

        # Put the cnn model in training mode
        cnn_model.train()

        # Create the combined function and structural optimization
        # setup

        # Calculate initial compliance
        initial_compliance, x_phys, _ = topo_physics.calculate_compliance(
            cnn_model, ke, args, device, default_dtype
        )
        initial_compliance = (
            torch.ceil(initial_compliance.to(torch.float64).detach()) + 1.0
        )
        initial_volume = torch.mean(x_phys)
        trials_initial_volumes.append(initial_volume.detach().cpu().numpy())

        # Combined function
        comb_fn = lambda model: pygranso_combined_function(  # noqa
            cnn_model,
            initial_compliance,
            ke,
            args,
            device=device,
            dtype=default_dtype,
        )

        # Initalize the pygranso options
        opts = pygransoStruct()

        # Set the device
        opts.torch_device = device

        # Setup the intitial inputs for the solver
        nvar = getNvarTorch(cnn_model.parameters())
        opts.x0 = (
            torch.nn.utils.parameters_to_vector(cnn_model.parameters())
            .detach()
            .reshape(nvar, 1)
        ).to(device=device, dtype=dtype32)

        # Additional pygranso options
        opts.limited_mem_size = 20
        opts.torch_device = device
        opts.double_precision = True
        opts.mu0 = mu
        opts.maxit = maxit
        opts.print_frequency = 1
        opts.stat_l2_model = False
        opts.viol_eq_tol = 1e-6
        opts.opt_tol = 1e-6

        mHLF_obj = utils.HaltLog()
        halt_log_fn, get_log_fn = mHLF_obj.makeHaltLogFunctions(opts.maxit)

        #  Set PyGRANSO's logging function in opts
        opts.halt_log_fn = halt_log_fn

        # Main algorithm with logging enabled.
        start = time.time()
        soln = pygranso(var_spec=cnn_model, combined_fn=comb_fn, user_opts=opts)
        end = time.time()
        wall_time = end - start

        # GET THE HISTORY OF ITERATES
        # Even if an error is thrown, the log generated until the error can be
        # obtained by calling get_log_fn()
        log = get_log_fn()

        # # Final structure
        # indexes = (pd.Series(log.fn_evals).cumsum() - 1).values.tolist()

        cnn_model.eval()
        with torch.no_grad():
            _, final_design, _ = topo_physics.calculate_compliance(
                cnn_model, ke, args, device, default_dtype
            )
            final_design = final_design.detach().cpu().numpy()

        # Put back metrics on original scale
        final_f = soln.final.f * initial_compliance.cpu().numpy()
        log_f = pd.Series(log.f) * initial_compliance.cpu().numpy()

        # Save the data from each trial
        fig = None
        if neptune_logging is not None:
            for f_value in log.f:
                neptune_logging[f"trial = {index} / loss"].log(f_value)

            best_score = np.round(final_f, 2)
            fig = utils.build_final_design(
                problem.name,
                final_design,
                best_score,
                requires_flip,
                total_frames,
                figsize=(10, 6),
            )
            neptune_logging[f"trial={index}-{problem.name}-final-design"].upload(fig)
            plt.close()

        # trials
        trials_designs[index, :, :] = final_design
        trials_losses[: len(log_f), index] = log_f.values  # noqa

        # Remove all variables for the next round
        del (
            cnn_model,
            comb_fn,
            opts,
            mHLF_obj,
            halt_log_fn,
            get_log_fn,
            soln,
            log,
            final_design,
            fig,
            final_f,
            log_f,
        )
        gc.collect()
        torch.cuda.empty_cache()

    outputs = {
        "designs": trials_designs,
        "losses": trials_losses,
        # Convert to numpy array
        "trials_initial_volumes": np.array(trials_initial_volumes),
    }

    return outputs


def train_multi_material_pygranso(
    problem,
    pygranso_combined_function,
    num_materials,
    device,
    requires_flip,
    total_frames,
    cnn_kwargs=None,
    neptune_logging=None,
    *,
    num_trials=50,
    mu=1.0,
    maxit=500,
) -> tuple:
    """
    Function to train structural optimization pygranso
    """
    # Set up the dtypes
    dtype32 = torch.float32
    default_dtype = utils.DEFAULT_DTYPE

    # Get the problem args
    args = topo_api.specified_task(problem, device=device)

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"],
        poisson=args["poisson"],
        device=device,
    )

    # Trials
    trials_designs = np.zeros((num_trials, num_materials, args["nely"], args["nelx"]))
    trials_losses = np.full((maxit + 1, num_trials), np.nan)
    trials_initial_volumes = []

    for index, seed in enumerate(range(0, num_trials)):
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize the CNN Model
        if cnn_kwargs is not None:
            cnn_model = models.MultiMaterialModel(args, **cnn_kwargs).to(
                device=device, dtype=dtype32
            )
        else:
            cnn_model = models.MultiMaterialModel(args).to(device=device, dtype=dtype32)

        # Put the cnn model in training mode
        cnn_model.train()

        # Create the combined function and structural optimization
        # setup

        # Calculate initial compliance
        (
            initial_compliance,
            x_phys,
            _,
        ) = topo_physics.calculate_multi_material_compliance(
            cnn_model, ke, args, device, default_dtype
        )
        initial_compliance = (
            torch.ceil(initial_compliance.to(torch.float64).detach()) + 1.0
        )
        initial_volume = torch.mean(x_phys)
        trials_initial_volumes.append(initial_volume.detach().cpu().numpy())

        # Combined function
        comb_fn = lambda model: pygranso_combined_function(  # noqa
            cnn_model,
            initial_compliance,
            ke,
            args,
            device=device,
            dtype=default_dtype,
        )

        # Initalize the pygranso options
        opts = pygransoStruct()

        # Set the device
        opts.torch_device = device

        # Setup the intitial inputs for the solver
        nvar = getNvarTorch(cnn_model.parameters())
        opts.x0 = (
            torch.nn.utils.parameters_to_vector(cnn_model.parameters())
            .detach()
            .reshape(nvar, 1)
        ).to(device=device, dtype=dtype32)

        # Additional pygranso options
        opts.limited_mem_size = 20
        opts.torch_device = device
        opts.double_precision = True
        opts.mu0 = mu
        opts.maxit = maxit
        opts.print_frequency = 1
        opts.stat_l2_model = False
        opts.viol_eq_tol = 1e-6
        opts.opt_tol = 1e-6

        mHLF_obj = utils.HaltLog()
        halt_log_fn, get_log_fn = mHLF_obj.makeHaltLogFunctions(opts.maxit)

        #  Set PyGRANSO's logging function in opts
        opts.halt_log_fn = halt_log_fn

        # Main algorithm with logging enabled.
        start = time.time()
        soln = pygranso(var_spec=cnn_model, combined_fn=comb_fn, user_opts=opts)
        end = time.time()
        wall_time = end - start

        # GET THE HISTORY OF ITERATES
        # Even if an error is thrown, the log generated until the error can be
        # obtained by calling get_log_fn()
        log = get_log_fn()

        # # Final structure
        # indexes = (pd.Series(log.fn_evals).cumsum() - 1).values.tolist()

        cnn_model.eval()
        with torch.no_grad():
            _, final_design, _ = topo_physics.calculate_compliance(
                cnn_model, ke, args, device, default_dtype
            )
            final_design = final_design.detach().cpu().numpy()

        # Put back metrics on original scale
        final_f = soln.final.f * initial_compliance.cpu().numpy()
        log_f = pd.Series(log.f) * initial_compliance.cpu().numpy()

        # Save the data from each trial
        fig = None
        if neptune_logging is not None:
            for f_value in log.f:
                neptune_logging[f"trial = {index} / loss"].log(f_value)

            best_score = np.round(final_f, 2)
            fig = utils.build_final_design(
                problem.name,
                final_design,
                best_score,
                requires_flip,
                total_frames,
                figsize=(10, 6),
            )
            neptune_logging[f"trial={index}-{problem.name}-final-design"].upload(fig)
            plt.close()

        # trials
        trials_designs[index, :, :] = final_design
        trials_losses[: len(log_f), index] = log_f.values  # noqa

        # Remove all variables for the next round
        del (
            cnn_model,
            comb_fn,
            opts,
            mHLF_obj,
            halt_log_fn,
            get_log_fn,
            soln,
            log,
            final_design,
            fig,
            final_f,
            log_f,
        )
        gc.collect()
        torch.cuda.empty_cache()

    outputs = {
        "designs": trials_designs,
        "losses": trials_losses,
        # Convert to numpy array
        "trials_initial_volumes": np.array(trials_initial_volumes),
    }

    return outputs


def train_adam(problem, cnn_kwargs=None, lr=4e-4, iterations=500):
    """
    Function that will train the structural optimization with
    the Adam optimizer
    """
    np.random.seed(0)

    # Get problem specific arguments
    args = topo_api.specified_task(problem)

    # Initiate the model to be trained
    # Current, assumption is a CNN model
    if cnn_kwargs is not None:
        model = models.CNNModel(args=args, **cnn_kwargs)
    else:
        model = models.CNNModel(args=args)

    # Build the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"], poisson=args["poisson"]
    )  # noqa

    # Set up the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # We want to save the frames and losses for running
    # and looking at tasks
    frames = []
    displacement_frames = []
    u_matrix_frames = []
    losses = []

    # Set up additional kwargs
    kwargs = dict(
        penal=torch.tensor(args["penal"]),
        e_min=torch.tensor(args["young_min"]),
        e_0=torch.tensor(args["young"]),
    )

    # Put the model in training mode
    model.train()

    # Train the model
    for iteration, i in enumerate(range(iterations)):
        # Zero out the gradients
        optimizer.zero_grad()

        # Get the model outputs
        logits = model(None)

        # Calculate the physical density
        x_phys = topo_physics.physical_density(
            logits, args, volume_constraint=True
        )  # noqa

        # Calculate the forces
        forces = topo_physics.calculate_forces(x_phys, args)

        # Calculate the u_matrix
        u_matrix = topo_physics.sparse_displace(
            x_phys, ke, args, forces, args["freedofs"], args["fixdofs"], **kwargs
        )

        # Calculate the compliance output
        compliance_output, _, _ = topo_physics.compliance(
            x_phys, u_matrix, ke, args, **kwargs
        )

        # The loss is the sum of the compliance
        loss = torch.sum(compliance_output)

        # Append the frames
        frames.append(logits)

        # Print the progress every 10 iterations
        if (iteration % 10) == 0:
            print(f"Compliance loss = {loss.item()} / Iteration={iteration}")
            losses.append(loss.item())

        # Go through the backward pass and create the gradients
        loss.backward()

        # Step through the optimzer to update the data with the gradients
        optimizer.step()

    # Render was also used in the original code to create
    # images of the structures
    render = [
        topo_physics.physical_density(x, args, volume_constraint=True)
        for x in frames  # noqa
    ]
    return render, losses


def train_lbfgs(problem, cnn_kwargs=None, lr=4e-4, iterations=500):
    """
    Function that will train the structural optimization with
    the Adam optimizer
    """
    # Get problem specific arguments
    args = topo_api.specified_task(problem)

    # Initiate the model to be trained
    # Current, assumption is a CNN model
    if cnn_kwargs is not None:
        model = models.CNNModel(args=args, **cnn_kwargs)
    else:
        model = models.CNNModel(args=args)

    # Build the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"], poisson=args["poisson"]
    )  # noqa

    # Set up the Adam optimizer
    optimizer = optim.LBFGS(model.parameters(), lr=lr)

    # We want to save the frames and losses for running
    # and looking at tasks
    frames = []
    displacement_frames = []
    u_matrix_frames = []
    losses = []

    # Set up additional kwargs
    kwargs = dict(
        penal=torch.tensor(args["penal"]),
        e_min=torch.tensor(args["young_min"]),
        e_0=torch.tensor(args["young"]),
    )

    # Put the model in training mode
    model.train()

    # Train the model
    for iteration, i in enumerate(range(iterations)):

        def closure():
            # Zero out the gradients
            optimizer.zero_grad()

            # Get the model outputs
            logits = model(None)

            # Calculate the physical density
            x_phys = topo_physics.physical_density(
                logits, args, volume_constraint=True
            )  # noqa

            # Calculate the forces
            forces = topo_physics.calculate_forces(x_phys, args)

            # Calculate the u_matrix
            u_matrix = topo_physics.sparse_displace(
                x_phys, ke, args, forces, args["freedofs"], args["fixdofs"], **kwargs
            )

            # Calculate the compliance output
            compliance_output, _, _ = topo_physics.compliance(
                x_phys, u_matrix, ke, args, **kwargs
            )  # noqa

            # The loss is the sum of the compliance
            loss = torch.sum(compliance_output)

            # Append the frames
            frames.append(logits)
            # displacement_frames.append(displacement)
            # u_matrix_frames.append(u_matrix)

            # Print the progress every 10 iterations
            if (iteration % 1) == 0:
                print(f"Compliance loss = {loss.item()} / Iteration={iteration}")
                losses.append(loss.item())

            # Go through the backward pass and create the gradients
            loss.backward()

            return loss

        # Step through the optimzer to update the data with the gradients
        optimizer.step(closure)

    # Render was also used in the original code to create
    # images of the structures
    render = [
        topo_physics.physical_density(x, args, volume_constraint=True)
        for x in frames  # noqa
    ]
    return render, losses


def train_google(
    problem, max_iterations=1000, cnn_kwargs=None, num_trials=50, neptune_logging=None
):
    """
    Replica of the google neural structural optimization training
    function in google colab
    """
    args = google_api.specified_task(problem)
    if cnn_kwargs is None:
        cnn_kwargs = {}

    trials = []
    for index, seed in enumerate(range(0, num_trials)):
        print(f"Google training trial {index + 1}")
        # Set seeds for this training
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Set up the model
        model = google_models.CNNModel(args=args, **cnn_kwargs)
        ds_cnn = google_train.train_lbfgs(model, max_iterations)

        dims = pd.Index(["google-cnn-lbfgs"], name="model")
        ds = xarray.concat([ds_cnn], dim=dims)

        # Extract the loss
        loss_df = ds.loss.transpose().to_pandas().cummin()
        loss_df = loss_df.reset_index(drop=True)
        loss_df = loss_df.rename_axis(index=None, columns=None)

        # Final loss
        final_loss = np.round(loss_df.min().values[0], 2)

        # Extract the image
        design = ds.design.sel(step=max_iterations, method="nearest").data.squeeze()
        design = design.astype(np.float16)

        if neptune_logging is not None:
            fig = utils.build_final_design(
                problem.name, design, final_loss, figsize=(10, 6)
            )
            neptune_logging[f"google-trial={index}-{problem.name}-final-design"].upload(
                fig
            )
            plt.close()

        # Append to the trials
        trials.append((final_loss, loss_df, design, None))

        del model, ds_cnn, dims, ds, loss_df, design
        gc.collect()

    return trials


def unconstrained_structural_optimization_function(model, ke, args, designs, losses):
    """
    Combined function for PyGranso for the structural optimization
    problem. The inputs will be the model that reparameterizes x as a function
    of a neural network. V0 is the initial volume, K is the global stiffness
    matrix and F is the forces that are applied in the problem.
    """
    # Initialize the model
    # In my version of the model it follows the similar behavior of the
    # tensorflow repository and only needs None to initialize and output
    # a first value of x
    logits = model(None)

    # kwargs for displacement
    kwargs = dict(
        penal=torch.tensor(args["penal"]),
        e_min=torch.tensor(args["young_min"]),
        e_0=torch.tensor(args["young"]),
    )

    # Calculate the physical density
    x_phys = topo_physics.physical_density(logits, args, volume_constraint=True)  # noqa

    # Calculate the forces
    forces = topo_physics.calculate_forces(x_phys, args)

    # Calculate the u_matrix
    u_matrix, _ = topo_physics.sparse_displace(
        x_phys, ke, forces, args["freedofs"], args["fixdofs"], **kwargs
    )

    # Calculate the compliance output
    compliance_output = topo_physics.compliance(x_phys, u_matrix, ke, **kwargs)

    # The loss is the sum of the compliance
    f = torch.sum(compliance_output)

    # Run this problem with no inequality constraints
    ci = None

    # Run this problem with no equality constraints
    ce = None

    # Append updated physical density designs
    designs.append(
        topo_physics.physical_density(logits, args, volume_constraint=True)
    )  # noqa

    return f, ci, ce


def train_constrained_adam(problem, cnn_kwargs=None, lr=4e-4, iterations=500):
    """
    Function that will train the structural optimization with
    the Adam optimizer
    """
    # Get problem specific arguments
    args = topo_api.specified_task(problem)

    # Initiate the model to be trained
    # Current, assumption is a CNN model
    if cnn_kwargs is not None:
        model = models.CNNModel(args=args, **cnn_kwargs)
    else:
        model = models.CNNModel(args=args)

    # Build the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"], poisson=args["poisson"]
    )  # noqa

    # Beta
    beta = list(model.parameters())[0]

    # Set up the Adam optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # We want to save the frames and losses for running
    # and looking at tasks
    frames = []
    compliance_losses = []
    volume_losses = []
    combined_losses = []

    # Set up additional kwargs
    kwargs = dict(
        penal=torch.tensor(args["penal"]),
        e_min=torch.tensor(args["young_min"]),
        e_0=torch.tensor(args["young"]),
    )

    # Put the model in training mode
    model.train()

    # Train the model
    for iteration, i in enumerate(range(iterations)):
        # Zero out the gradients
        optimizer.zero_grad()

        # Get the model outputs
        logits = model(None)

        # Instead of calculating x_phys directly
        # we will see if we can add it to the objective
        # function and learn it this way
        x_phys = torch.sigmoid(logits + torch.mean(beta))

        # Calculate the forces
        forces = topo_physics.calculate_forces(x_phys, args)

        # Calculate the u_matrix
        u_matrix, _ = topo_physics.sparse_displace(
            x_phys, ke, forces, args["freedofs"], args["fixdofs"], **kwargs
        )

        # Calculate the compliance output
        compliance_output = topo_physics.compliance(
            x_phys, u_matrix, ke, **kwargs
        )  # noqa

        # The loss is the sum of the compliance
        compliance_loss = torch.sum(compliance_output)
        volume_loss = 1e3 * torch.abs(torch.mean(x_phys) - args["volfrac"])
        loss = compliance_loss + volume_loss

        # Append the frames
        frames.append(x_phys)

        # Print the progress every 10 iterations
        if (iteration % 10) == 0:
            print(f"loss = {loss.item()} / Iteration={iteration}")
            print(f"compliance_loss = {compliance_loss.item()}")
            print(f"volume loss = {volume_loss.item()}")
            print(f"Mean of physical density = {torch.mean(x_phys)}")
            print("")
            combined_losses.append(loss.item())
            compliance_losses.append(compliance_loss.item())
            volume_losses.append(compliance_loss.item())

        # Go through the backward pass and create the gradients
        loss.backward()

        # Step through the optimzer to update the data with the gradients
        optimizer.step()

    # Render was also used in the original code to create
    # images of the structures
    render = [x for x in frames]  # noqa

    # Combined losses
    losses = {
        "combined_losses": combined_losses,
        "compliance_losses": compliance_losses,
        "volume_losses": volume_losses,
    }
    return render, losses, u_matrix
