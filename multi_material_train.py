# stdlib
import gc

# third party
import numpy as np
import torch
from pygranso.private.getNvar import getNvarTorch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

# first party
import models
import topo_api
import topo_physics


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
    unscaled_compliance, full_x_phys = topo_physics.calculate_multi_material_compliance(
        model, ke, args, device, dtype
    )
    f = 1.0 / initial_compliance * unscaled_compliance

    # Run this problem with no inequality constraints
    ci = None

    ce = pygransoStruct()
    total_mass = (
        args["combined_volfrac"] * args["nelx"] * args["nely"] * np.max(args["volfrac"])
    )

    x_phys = full_x_phys[1:, :, :]

    x_phys_mass = torch.zeros(len(args["volfrac"]))
    for index, density in enumerate(args["volfrac"]):
        x_phys_mass[index] = density * torch.sum(x_phys[index, :, :])

    ce.c1 = (torch.sum(x_phys_mass) / total_mass) - 1.0  # noqa

    # # Let's try and clear as much stuff as we can to preserve memory
    del x_phys, ke
    gc.collect()
    torch.cuda.empty_cache()

    return f, ci, ce


def train_mass_constrained_multi_material(
    problem,
    model_type,
    e_materials,
    num_materials,
    volfrac,
    combined_volfrac,
    penal,
    seed=0,
):
    """
    Function to train the mass constrained multi-material problem with
    pygranso
    """
    # Set up seed for reproducibility
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For now setup the device to be CPU
    device = torch.device("cpu")

    # Get the args
    args = topo_api.specified_task(problem, device=device)

    # TODO: Once we get this example working we can configure the problem in the right
    # way but for now we will set it from the inputs to the function
    args = topo_api.specified_task(problem, device=device)
    args["e_materials"] = e_materials
    args["num_materials"] = num_materials
    args["volfrac"] = volfrac
    args["combined_volfrac"] = combined_volfrac
    args["penal"] = penal

    # Get the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"],
        poisson=args["poisson"],
        device=device,
    ).to(dtype=torch.double)

    # Keyword arguments for the different types of models we will
    # allow. Currently, we have the DIP multi-material and the MLP
    # multi-material for comparison
    numLayers = 5  # noqa
    numNeuronsPerLyr = 20  # noqa
    cnn_kwargs = dict(resizes=(1, 2, 2, 2, 1))

    # Select the model
    if model_type == "cnn":
        model = models.MultiMaterialModel(args, **cnn_kwargs).to(
            device=device, dtype=torch.double
        )

    elif model_type == "mlp":
        model = models.TopNet(
            numLayers,
            numNeuronsPerLyr,
            args["nelx"],
            args["nely"],
            args["num_materials"],
            symXAxis=False,
            symYAxis=False,
        ).to(device=device, dtype=torch.double)

    else:
        raise ValueError("There is no such model!")

    # Put the model in training mode
    model.train()

    # Calculate the inital compliance
    initial_compliance, x_phys = topo_physics.calculate_multi_material_compliance(
        model, ke, args, device, torch.double
    )
    initial_compliance = torch.ceil(initial_compliance.to(torch.float64).detach()) + 1.0

    # Combined function
    comb_fn = lambda model: multi_material_volume_constrained_structural_optimization_function(  # noqa
        model,
        initial_compliance,
        ke,
        args,
        device=device,
        dtype=torch.double,
    )

    # Initalize the pygranso options
    opts = pygransoStruct()

    # Set the device
    opts.torch_device = device

    # Setup the intitial inputs for the solver
    nvar = getNvarTorch(model.parameters())
    opts.x0 = (
        torch.nn.utils.parameters_to_vector(model.parameters())
        .detach()
        .reshape(nvar, 1)
    ).to(device=device)

    # Additional pygranso options
    opts.limited_mem_size = 20
    opts.torch_device = device
    opts.double_precision = True
    opts.mu0 = 1.0
    opts.maxit = 100
    opts.print_frequency = 1
    opts.stat_l2_model = False
    opts.viol_eq_tol = 1e-4
    opts.opt_tol = 1e-4
    opts.init_step_size = 1e-1

    # Main algorithm with logging enabled.
    soln = pygranso(var_spec=model, combined_fn=comb_fn, user_opts=opts)

    return model, ke, args, soln
