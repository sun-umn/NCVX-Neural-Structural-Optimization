import time

import torch
import torch.nn as nn
import torch.optim as optim
from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

import models
import topo_api
import topo_physics
import utils


def train_adam(problem, cnn_kwargs=None, lr=4e-4, iterations=500):
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
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # We want to save the frames and losses for running
    # and looking at tasks
    frames = []
    displacement_frames = []
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
        u_matrix, _ = topo_physics.sparse_displace(
            x_phys, ke, forces, args["freedofs"], args["fixdofs"], **kwargs
        )

        # Calculate the compliance output
        compliance_output, displacement = topo_physics.compliance(
            x_phys, u_matrix, ke, **kwargs
        )  # noqa

        # The loss is the sum of the compliance
        loss = torch.sum(compliance_output)

        # Append the frames
        frames.append(logits)
        displacement_frames.append(displacement)

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
    return render, losses, displacement_frames


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


def train_pygranso(
    problem,
    cnn_kwargs=None,
    *,
    device="cpu",
    mu=1.0,
    maxit=150,
    init_step_size=5e-6,
    linesearch_maxit=50,
    linesearch_reattempts=15,
) -> tuple:
    """
    Function to train structural optimization pygranso
    """
    # Get the problem args
    args = topo_api.specified_task(problem)

    # Initialize the CNN Model
    if cnn_kwargs is not None:
        cnn_model = models.CNNModel(args, **cnn_kwargs)
    else:
        cnn_model = models.CNNModel(args)

    # Put the cnn model in training mode
    cnn_model.train()

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"],
        poisson=args["poisson"],
    )

    # Create the combined function and structural optimization
    # setup
    # Save the physical density designs & the losses
    designs = []
    losses = []
    # Combined function
    comb_fn = lambda model: unconstrained_structural_optimization_function(  # noqa
        model, ke, args, designs, losses
    )

    # Initalize the pygranso options
    opts = pygransoStruct()

    # Set the device
    opts.torch_device = torch.device(device)

    # Setup the intitial inputs for the solver
    nvar = getNvarTorch(cnn_model.parameters())
    opts.x0 = (
        torch.nn.utils.parameters_to_vector(cnn_model.parameters())
        .detach()
        .reshape(nvar, 1)
    )

    # Additional pygranso options
    opts.limited_mem_size = 20
    opts.double_precision = True
    opts.mu0 = mu
    opts.maxit = maxit
    opts.print_frequence = 10
    opts.stat_l2_model = False

    # Other parameters that helped the structural optimization
    # problem
    opts.init_step_size = init_step_size
    opts.linesearch_maxit = linesearch_maxit
    opts.linesearch_reattempts = linesearch_reattempts

    # Save the end results using the halt function
    halt_function_obj = utils.HaltLog()
    halt_log_fn, get_log_fn = halt_function_obj.makeHaltLogFunctions(opts.maxit)

    #  Set PyGRANSO's logging function in opts
    opts.halt_log_fn = halt_log_fn

    # Train pygranso
    start = time.time()
    soln = pygranso(var_spec=cnn_model, combined_fn=comb_fn, user_opts=opts)
    end = time.time()

    # After pygranso runs we can gather the logs
    log = get_log_fn()

    # Print time
    print(f"Total wall time: {end - start} seconds")

    return soln, designs, log


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


def train_u_matrix_adam(problem, cnn_kwargs=None, lr=4e-4, iterations=500):
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

    # Get u_matrix
    u_nonzero = list(model.parameters())[0]
    neural_network_model_parameters = list(model.parameters())[1:]

    # Get free and fixed degrees of freedom
    freedofs = args["freedofs"]
    fixdofs = args["fixdofs"]
    index_map = torch.cat((freedofs, fixdofs))
    index_map = torch.argsort(index_map)

    # Build the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"], poisson=args["poisson"]
    )  # noqa

    # Set up the Adam optimizer
    parameters = [
        {"params": [u_nonzero], "lr": 5e-1},
        {"params": neural_network_model_parameters, "lr": 4e-4},
    ]
    optimizer = optim.Adam(parameters)
    # scheduler = (
    #     optim
    #     .lr_scheduler
    #     .ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5)
    # )

    # We want to save the frames and losses for running
    # and looking at tasks
    frames = []
    displacement_frames = []
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
        node_indexes = utils.build_node_indexes(x_phys)
        forces_nodes = forces[node_indexes]

        # Calculate the full u
        u = torch.cat((u_nonzero, torch.zeros(len(fixdofs))))
        u = u[index_map]

        # Calculate the compliance output
        compliance_output, displacement, ke_u = topo_physics.compliance(
            x_phys, u, ke, **kwargs
        )  # noqa

        # Equilibrium constraint
        alpha = torch.mean(torch.pow(u, 2))
        equilibrium_loss = torch.mean(torch.pow((forces_nodes - ke_u), 2))

        # # The loss is the sum of the compliance
        loss = torch.sum(compliance_output) + equilibrium_loss + 1e3 / alpha
        # loss = equilibrium_loss

        # Append the frames
        frames.append(logits)
        displacement_frames.append(displacement)

        # Print the progress every 10 iterations
        if (iteration % 100) == 0:
            print(f"loss = {loss.item()} / Iteration={iteration}")
            print(f"compliance output = {torch.sum(compliance_output)}")
            print(f"Equilibrium loss = {equilibrium_loss}")
            losses.append(loss.item())

        # Go through the backward pass and create the gradients
        loss.backward()

        # Step through the optimzer to update the data with the gradients
        optimizer.step()
        # scheduler.step(loss)

    # Render was also used in the original code to create
    # images of the structures
    render = [
        topo_physics.physical_density(x, args, volume_constraint=True)
        for x in frames  # noqa
    ]

    outputs = {
        "u": u,
        "ke_u": ke_u,
        "forces_nodes": forces_nodes,
    }
    return render, losses, displacement_frames, outputs


def train_u_matrix(problem, iterations=20000):
    """
    Function that will train the structural optimization with
    the Adam optimizer
    """
    # Get problem specific arguments
    args = topo_api.specified_task(problem)

    # Initiate the model to be trained
    # In this case the u matrix model
    model = models.UMatrixModel(
        args=args,
        uniform_lower_bound=-5500,
        uniform_upper_bound=5500,
    )

    # Get free and fixed degrees of freedom
    freedofs = args["freedofs"]
    fixdofs = args["fixdofs"]
    index_map = torch.cat((freedofs, fixdofs))
    index_map = torch.argsort(index_map)

    # Build the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"], poisson=args["poisson"]
    ).double()  # noqa

    # Set up the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=5e-1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=100, factor=0.5
    )

    # We want to save the frames and losses for running
    # and looking at tasks
    frames = []
    losses = []

    # Put the model in training mode
    model.train()

    # Train the model
    for iteration, i in enumerate(range(iterations)):
        # Zero out the gradients
        optimizer.zero_grad()

        # Get the model outputs
        u_nonzero = model(None)

        # Calculate the physical density
        x_phys = torch.zeros(
            args["nely"],
            args["nelx"],
        )

        # Calculate the forces
        forces = topo_physics.calculate_forces(x_phys, args)
        node_indexes = utils.build_node_indexes(x_phys)
        forces_nodes = forces[node_indexes]

        # Calculate the full u
        u = torch.cat((u_nonzero, torch.zeros(len(fixdofs))))
        u = u[index_map].double()
        displacement_field = u[node_indexes].squeeze()

        # Run the compliance calculation
        ke_u = torch.einsum("ij,jkl->ikl", ke, displacement_field)

        # Equilibrium constraint
        loss = torch.mean(torch.pow((forces_nodes - ke_u), 2))

        # Print the progress every 10 iterations
        if (iteration % 1000) == 0:
            losses.append(loss.item())

        # Go through the backward pass and create the gradients
        loss.backward()

        # Step through the optimzer to update the data with the gradients
        optimizer.step()
        scheduler.step(loss)

    print(f"    equilibrium loss = {loss.item()} / Iteration={iteration}")
    return u.detach(), displacement_field


def train_u_full_k_matrix(
    problem, model, x_phys=None, iterations=20000, lr=5e-1, logging=False, warmup=False
):
    """
    Function that will train the structural optimization with
    the Adam optimizer
    """
    # Get problem specific arguments
    args = topo_api.specified_task(problem)

    # Calculate the physical density
    if x_phys is None:
        x_phys = torch.ones(args["nely"], args["nelx"]) * 0.5
    else:
        x_phys = x_phys.detach()

    # Build the full K which is dependent on x_phys
    K = topo_physics.build_full_K_matrix(
        x_phys,
        args,
    )

    # Get free and fixed degrees of freedom
    freedofs = args["freedofs"]
    fixdofs = args["fixdofs"]
    index_map = torch.cat((freedofs, fixdofs))
    index_map = torch.argsort(index_map)

    # Build the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"], poisson=args["poisson"]
    ).double()  # noqa

    # Get K freedofs
    K_freedofs = K[:, freedofs][freedofs, :]

    # Set up the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # We want to save the frames and losses for running
    # and looking at tasks
    frames = []
    losses = []

    # Put the model in training mode
    model.train()

    # Train the model
    for iteration, i in enumerate(range(iterations)):
        # Zero out the gradients
        optimizer.zero_grad()

        # Get the model outputs
        u_nonzero = model(None)

        # Calculate the forces
        forces = topo_physics.calculate_forces(x_phys, args)
        forces_freedofs = forces[freedofs]
        node_indexes = utils.build_node_indexes(x_phys)
        forces_nodes = forces[node_indexes].squeeze()

        # Calculate the full u
        u = torch.cat((u_nonzero, torch.zeros(len(fixdofs))))
        u = u[index_map].reshape(len(u), 1).double()
        displacement_field = u[node_indexes].squeeze()

        # Run the compliance calculation
        ke_u = torch.einsum("ij,jkl->ikl", ke, displacement_field)

        # # Equilibrium constraint
        # KU = torch.einsum('ij,jk->ik', K, u).flatten()
        KU = torch.einsum(
            "ij,ik->ik", K_freedofs, u_nonzero.reshape(len(u_nonzero), 1)
        ).flatten()

        if warmup:
            loss = torch.mean(torch.pow((forces_nodes - ke_u), 2))
        else:
            loss = torch.mean(torch.pow(forces_freedofs - KU, 2))

        # Print the progress every 10 iterations
        if (iteration % 100) == 0:
            if logging:
                print(f"    equilibrium loss = {loss.item()} / Iteration={iteration}")
            losses.append(loss.item())

        # Go through the backward pass and create the gradients
        loss.backward()

        # Step through the optimzer to update the data with the gradients
        optimizer.step()

    print(f"    equilibrium loss = {loss.item()} / Iteration={iteration}")
    return u.flatten().detach(), displacement_field


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

    # Get free and fixed degrees of freedom
    freedofs = args["freedofs"]
    fixdofs = args["fixdofs"]
    index_map = torch.cat((freedofs, fixdofs))
    index_map = torch.argsort(index_map)

    # Build the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"], poisson=args["poisson"]
    )  # noqa

    # Set up the Adam optimizer
    optimizer = optim.Adam(model.parameters())

    # We want to save the frames and losses for running
    # and looking at tasks
    frames = []
    displacement_frames = []
    losses = []

    # Set up additional kwargs
    kwargs = dict(
        penal=torch.tensor(args["penal"]),
        e_min=torch.tensor(args["young_min"]),
        e_0=torch.tensor(args["young"]),
    )

    # Put the model in training mode
    model.train()

    # U matrix model
    u_matrix_model = models.UMatrixModel(
        args=args,
        uniform_lower_bound=-5500,
        uniform_upper_bound=5500,
    )
    u_matrix_model.train()

    # Warm up training for u
    print("u matrix warmup")
    u, displacement_field = train_u_full_k_matrix(
        problem,
        u_matrix_model,
        x_phys=None,
        iterations=20000,
        logging=True,
        warmup=True,
    )
    displacement_frames.append(displacement_field)

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

        # Calculate U matrix
        u, displacement_field = train_u_full_k_matrix(
            problem, u_matrix_model, x_phys, iterations=500, logging=True, lr=5e-4
        )
        u = 1e2 * u

        # Calculate the compliance output
        compliance_output, displacement, ke_u = topo_physics.compliance(
            x_phys, u, ke, **kwargs
        )  # noqa

        # # The loss is the sum of the compliance
        loss = torch.sum(compliance_output)

        # Append the frames
        frames.append(logits)
        displacement_frames.append(displacement_field)

        # Print the progress every 10 iterations
        if (iteration % 1) == 0:
            print(f"loss = {loss.item()} / Iteration={iteration}")
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

    return render, losses, displacement_frames
