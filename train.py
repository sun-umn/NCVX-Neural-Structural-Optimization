import torch
import torch.optim as optim

import models
import topo_api
import topo_physics
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch


def train_adam(problem, cnn_kwargs, lr=4e-4, iterations=500):
    """
    Function that will train the structural optimization with
    the Adam optimizer
    """
    # Get problem specific arguments
    args = topo_api.specified_task(problem)

    # Initiate the model to be trained
    # Current, assumption is a CNN model
    model = models.CNNModel(args=args, **cnn_kwargs)

    # Build the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(young=args["young"], poisson=args["poisson"])

    # Set up the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # We want to save the frames and losses for running
    # and looking at tasks
    frames = []
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
        x_phys = topo_physics.physical_density(logits, args, volume_constraint=True)

        # Calculate the forces
        forces = topo_physics.calculate_forces(x_phys, args)

        # Calculate the u_matrix
        u_matrix = topo_physics.sparse_displace(
            x_phys, ke, forces, args["freedofs"], args["fixdofs"], **kwargs
        )

        # Calculate the compliance output
        compliance_output = topo_physics.compliance(x_phys, u_matrix, ke, **kwargs)

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

    # Render was also used in the original code to create images of the structures
    render = [
        topo_physics.physical_density(x, args, volume_constraint=True) for x in frames
    ]
    return render, losses


def structural_optimization_function(model, ke, args, designs, debug=False):
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
    x_phys = topo_physics.physical_density(logits, args, volume_constraint=True)
    
    # Calculate the forces
    forces = topo_physics.calculate_forces(x_phys, args)
    
    # Calculate the u_matrix
    u_matrix = topo_physics.sparse_displace(
        x_phys, ke, forces, args['freedofs'], args['fixdofs'], **kwargs
    )
    
    # Calculate the compliance output
    compliance_output = topo_physics.compliance(x_phys, u_matrix, ke, **kwargs)
    
    # The loss is the sum of the compliance
    f = torch.sum(compliance_output)
    
    # Run this problem with no inequality constraints
    ci = None
    
    # Run this problem with no equality constraints
    ce = None
    
    designs.append(topo_physics.physical_density(logits, args, volume_constraint=True))
    
    return f, ci, ce
