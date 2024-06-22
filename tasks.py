#!/usr/bin/python
# stdlib
import os
import pickle
import warnings

# third party
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import xarray
from neural_structural_optimization import models as google_models
from neural_structural_optimization import topo_api as google_topo_api
from neural_structural_optimization import train as google_train

# first party
import models
import problems
import topo_api
import topo_physics
import train
import utils
from MMTOuNN.neuralTO_MM import TopologyOptimizer as MMTO
from TOuNN.TOuNN import TopologyOptimizer

# Filter warnings
warnings.filterwarnings('ignore')

# NOTES:
# Keep these imports
# from matplotlib.offsetbox import AnchoredText
# from mpl_toolkits.axes_grid1 import make_axes_locatable


# Define the cli group
@click.group()
def cli():  # noqa
    pass


def calculate_mass_constraint(
    design, nelx, nely, material_density_weight, combined_frac
):
    """
    Compute the total mass constraint from a final design
    """
    design = np.array(design)

    material_density_weight = np.array(material_density_weight)
    total_mass = (
        np.max(material_density_weight)
        * design.shape[0]  # Expecting (nelx * nely, material) outputs
        * combined_frac
    )

    num_materials = len(material_density_weight)
    mass_constraint = np.zeros(num_materials)
    for index, density_weight in enumerate(material_density_weight):
        mass_constraint[index] = density_weight * np.sum(design[:, index + 1])

    return mass_constraint.sum() / total_mass - 1.0


def calculate_binary_constraint(design, mask, epsilon):
    """
    Function to compute the binary constraint
    """
    return np.round(np.mean(design[mask] * (1 - design[mask])) - epsilon, 4)


def calculate_volume_constraint(design, mask, volume):
    """
    Function that computes the volume constraint
    """
    return np.round(np.mean(design[mask]) / volume - 1.0, 4)


def build_outputs(problem_name, outputs, mask, volume, requires_flip, epsilon=1e-3):
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
    # Compute the binary and volume constraints
    binary_constraint = calculate_binary_constraint(
        design=best_final_design,
        mask=mask,
        epsilon=epsilon,
    )

    # volume constraint
    volume_constraint = calculate_volume_constraint(
        design=best_final_design,
        mask=mask,
        volume=volume,
    )

    if requires_flip:
        if (
            ("mbb" in problem_name)
            or ("l_shape" in problem_name)
            or ("cantilever" in problem_name)
        ):
            best_final_design = np.hstack(
                [best_final_design[:, ::-1], best_final_design]
            )

        elif (
            ("multistory" in problem_name)
            or ("thin" in problem_name)
            or ("michell" in problem_name)
        ):
            best_final_design = np.hstack(
                [best_final_design, best_final_design[:, ::-1]] * 2
            )

    # last row, first column (-1, 0)
    best_score = np.round(losses_df.iloc[-1, 0], 2)

    # Create metrics
    metrics = {
        'loss': outputs["losses"],
        'volume_constraint': outputs['volumes'],
        'binary_constraint': outputs['binary_constraint'],
        'symmetry_constraint': outputs['symmetry_constraint'],
    }

    return best_final_design, best_score, binary_constraint, volume_constraint, metrics


def build_google_outputs(
    problem_name, iterations, ds, mask, volume, requires_flip, epsilon=1e-3
):
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
        design=cnn_final_design, mask=mask, epsilon=epsilon
    )
    cnn_volume_constraint = calculate_volume_constraint(
        design=cnn_final_design,
        mask=mask,
        volume=volume,
    )

    # MMA final design
    mma_final_design = final_designs[1, :, :]
    mma_binary_constraint = calculate_binary_constraint(
        design=mma_final_design,
        mask=mask,
        epsilon=epsilon,
    )
    mma_volume_constraint = calculate_volume_constraint(
        design=mma_final_design,
        mask=mask,
        volume=volume,
    )

    if requires_flip:
        if (
            ("mbb" in problem_name)
            or ("l_shape" in problem_name)
            or ("cantilever" in problem_name)
        ):
            cnn_final_design = np.hstack([cnn_final_design[:, ::-1], cnn_final_design])
            mma_final_design = np.hstack([mma_final_design[:, ::-1], mma_final_design])

        if (
            ("multistory" in problem_name)
            or ("thin" in problem_name)
            or ("michell" in problem_name)
        ):
            cnn_final_design = np.hstack(
                [cnn_final_design, cnn_final_design[:, ::-1]] * 2
            )
            mma_final_design = np.hstack(
                [mma_final_design, mma_final_design[:, ::-1]] * 2
            )

    # Here compute the trajectories
    cnn_volume_constraint_trajectory = []
    cnn_binary_constraint_trajectory = []
    mma_volume_constraint_trajectory = []
    mma_binary_constraint_trajectory = []
    for i in range(iterations):
        # Get the intermediate designs
        design = ds.design.sel(step=i, method="nearest").data

        # CNN final design
        cnn_design = design[0, :, :]
        cnn_binary_constraint = calculate_binary_constraint(
            design=cnn_design, mask=mask, epsilon=epsilon
        )
        cnn_volume_constraint = calculate_volume_constraint(
            design=cnn_design,
            mask=mask,
            volume=volume,
        )
        cnn_volume_constraint_trajectory.append(cnn_volume_constraint)
        cnn_binary_constraint_trajectory.append(cnn_binary_constraint)

        # MMA final design
        mma_design = design[1, :, :]
        mma_binary_constraint = calculate_binary_constraint(
            design=mma_design,
            mask=mask,
            epsilon=epsilon,
        )
        mma_volume_constraint = calculate_volume_constraint(
            design=mma_design,
            mask=mask,
            volume=volume,
        )
        mma_volume_constraint_trajectory.append(mma_volume_constraint)
        mma_binary_constraint_trajectory.append(mma_binary_constraint)

    # Create the trajectories
    cnn_metrics = {
        'loss': losses["cnn-lbfgs"],
        'volume_constraint': cnn_volume_constraint_trajectory,
        'binary_constraint': cnn_binary_constraint_trajectory,
    }

    mma_metrics = {
        'loss': losses["mma"],
        'volume_constraint': mma_volume_constraint_trajectory,
        'binary_constraint': mma_binary_constraint_trajectory,
    }

    return {
        "google-cnn": (
            cnn_final_design,
            cnn_loss,
            cnn_binary_constraint,
            cnn_volume_constraint,
            cnn_metrics,
        ),
        "mma": (
            mma_final_design,
            mma_loss,
            mma_binary_constraint,
            mma_volume_constraint,
            mma_metrics,
        ),
    }


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


def tounn_train_and_outputs(problem, requires_flip):
    """
    Function that will run the TOuNN pipeline
    """
    # Try setting seed here as well
    models.set_seed(0)

    # Get the problem name
    problem_name = problem.name

    # Get the arguments for the problem
    args = topo_api.specified_task(problem)

    # Get the problem dimensions
    # Our arguments come as torch tensors so we need to convert back
    # to numpy and int to work with their framework
    nelx, nely = int(args['nelx'].numpy()), int(args['nely'].numpy())

    # Get the volume fraction
    desiredVolumeFraction = args['volfrac']

    # Get the forces - convert to numpy to work with
    # their pipeline
    force = args['forces'].cpu().numpy()

    # Add an extra axis because their framework expects
    # (n, 1)
    force = force[:, None]

    # Get the fixed dofs
    fixed = args['fixdofs'].cpu().numpy()

    # Get epsilon value
    epsilon = args['epsilon']

    # Get the mask for tounn problems
    tounn_mask = args['tounn_mask']

    # TODO: Figure out how this works with non-design
    # regions but for now it will be none
    nonDesignRegion = {
        'Rect': tounn_mask,
        'Circ': None,
        'Annular': None,
    }

    # Symmetry about axes
    symXAxis = False
    symYAxis = False

    # Penal in their code starts at 2
    penal = 2

    # Neural network config
    numLayers = 5
    numNeuronsPerLyr = 20
    minEpochs = 20
    maxEpochs = 1500
    useSavedNet = False

    # Run the pipeline
    topOpt = TopologyOptimizer()

    # Initialize the FE (Finite element) solver
    topOpt.initializeFE(
        problem.name, nelx, nely, force, fixed, penal, nonDesignRegion, args=args
    )

    # Initialize the optimizer
    topOpt.initializeOptimizer(
        numLayers, numNeuronsPerLyr, desiredVolumeFraction, symXAxis, symYAxis
    )

    # Run the optimization
    topOpt.optimizeDesign(maxEpochs, minEpochs, useSavedNet)

    # After everything is fitted we need to extract the final information
    # Set the plotResolution to 1
    plotResolution = 1

    # compute the points for the problem
    xyPlot, nonDesignPlotIdx = topOpt.generatePoints(
        topOpt.FE.nelx, topOpt.FE.nely, plotResolution, topOpt.nonDesignRegion
    )

    # Compute the final density
    density = torch.flatten(topOpt.topNet(xyPlot, nonDesignPlotIdx))
    density = density.detach().cpu().numpy()
    best_final_design = density.copy()

    # The final design needs to be reshaped and transposed
    best_final_design = best_final_design.reshape(
        plotResolution * topOpt.FE.nelx, plotResolution * topOpt.FE.nely
    )
    best_final_design = best_final_design.T

    # get the best score
    best_score = np.round(topOpt.convergenceHistory[-1][4], 2)

    # Return everything
    mask = (torch.broadcast_to(args["mask"], (nely, nelx)) > 0).cpu().numpy()

    # Compute the binary constraint
    binary_constraint = calculate_binary_constraint(
        design=best_final_design,
        mask=mask,
        epsilon=epsilon,
    )
    volume_constraint = calculate_volume_constraint(
        design=best_final_design,
        mask=mask,
        volume=desiredVolumeFraction,
    )

    # Add more information about the outputs
    if requires_flip:
        if (
            ("mbb" in problem_name)
            or ("l_shape" in problem_name)
            or ("cantilever" in problem_name)
        ):
            best_final_design = np.hstack(
                [best_final_design[:, ::-1], best_final_design]
            )

        if (
            ("multistory" in problem_name)
            or ("thin" in problem_name)
            or ("michell" in problem_name)
        ):
            best_final_design = np.hstack(
                [best_final_design, best_final_design[:, ::-1]] * 2
            )

    # Here will create a dict to save the losses
    metrics = {
        'loss': topOpt.convergenceHistory[4],
        'volume_constraint': topOpt.convergenceHistory[5],
        'binary_constraint': topOpt.convergenceHistory[6],
    }
    metrics = {
        'loss': np.array(
            [value for _, _, _, _, value, _, _ in topOpt.convergenceHistory]
        ),
        'volume_constraint': np.array(
            [value for _, _, _, _, _, value, _ in topOpt.convergenceHistory]
        ),
        'binary_constraint': np.array(
            [value for _, _, _, _, _, _, value in topOpt.convergenceHistory]
        ),
    }

    return best_final_design, best_score, binary_constraint, volume_constraint, metrics


def mmtounn_train_and_outputs(
    nelx, nely, e_materials, material_density_weight, combined_frac, seed=1234
):
    """
    Function that will run the TOuNN pipeline
    """
    # Set a different seed?

    elemArea = 1.0

    # Network config
    numLayers = 5
    # the depth of the NN
    numNeuronsPerLyr = 20
    # the height of the NN

    # problem
    exampleName = 'TipCantilever'

    args = topo_api.multi_material_tip_cantilever_task(
        nelx=nelx,
        nely=nely,
        e_materials=e_materials,
        material_density_weight=material_density_weight,
        combined_frac=combined_frac,
    )

    fixed = args['fixdofs'].numpy().astype(int)
    force = args['forces'].numpy().astype(np.float64)

    # e_materials and material_density_weight need to be numpy arrays
    e_materials = e_materials.numpy()
    material_density_weight = material_density_weight.numpy()

    nonDesignRegion = None
    symXAxis = False
    symYAxis = False

    # Additional config
    minEpochs = 50
    maxEpochs = 500
    penal = 1.0
    useSavedNet = False
    device = 'cpu'

    # Compute the topology
    topOpt = MMTO()
    topOpt.initializeFE(
        exampleName,
        nelx,
        nely,
        elemArea,
        force,
        fixed,
        device,
        penal,
        nonDesignRegion,
        e_materials,
    )
    topOpt.initializeOptimizer(
        numLayers=numLayers,
        numNeuronsPerLyr=numNeuronsPerLyr,
        desiredMassFraction=combined_frac,
        massDensityMaterials=material_density_weight,
        symXAxis=symXAxis,
        symYAxis=symYAxis,
        seed=seed,
    )

    # Run the optimization
    _ = topOpt.train(maxEpochs, minEpochs, useSavedNet)

    # Get the density outputs
    plotResolution = 1
    xyPlot, nonDesignPlotIdx = topOpt.generatePoints(
        topOpt.FE.nelx, topOpt.FE.nely, plotResolution, topOpt.nonDesignRegion
    )

    # Compute the final density
    density = topOpt.topNet(xyPlot, nonDesignPlotIdx)
    density = density.detach().cpu().numpy()

    return topOpt, density


@cli.command('run-multi-structure-pipeline')
@click.option('--model_size', default='medium')
@click.option('--structure_size', default='medium')
def run_multi_structure_pipeline(model_size, structure_size):
    """
    Task that will build out multiple structures and compare
    performance against known benchmarks.
    """

    # CNN parameters
    cnn_features = (256, 128, 64, 32, 16)
    kernel_size = (12, 12)

    # Configurations
    configs = {
        'tiny': {
            'latent_size': 96,
            'dense_channels': 24,
            'conv_filters': tuple(features // 6 for features in cnn_features),
        },
        'xsmall': {
            'latent_size': 96,
            'dense_channels': 24,
            'conv_filters': tuple(features // 5 for features in cnn_features),
        },
        'small': {
            'latent_size': 96,
            'dense_channels': 24,
            'conv_filters': tuple(features // 4 for features in cnn_features),
            'kernel_size': kernel_size,
        },
        'medium': {
            'latent_size': 96,
            'dense_channels': 24,
            'conv_filters': tuple(features // 3 for features in cnn_features),
            'kernel_size': kernel_size,
        },
        'large': {
            'latent_size': 96,
            'dense_channels': 24,
            'conv_filters': tuple(features // 2 for features in cnn_features),
            'kernel_size': kernel_size,
        },
        # x-large has been our original architecture
        'xlarge': {
            'latent_size': 96,
            'dense_channels': 24,
            'conv_filters': tuple(features // 1 for features in cnn_features),
        },
    }

    # CNN kwargs
    cnn_kwargs = configs[model_size]

    # Set seed
    models.set_seed(0)  # Model seed is set here but results are changing?

    # For testing we will run two experimentation trackers
    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Topology-Optimization'

    # Enable wandb
    wandb.login(key=API_KEY)

    # Initalize wandb
    # TODO: Save training and validation curves per fold
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=['topology-optimization-task', model_size, f'{structure_size}-structures'],
        config=cnn_kwargs,
    )

    # Will create directories for saving models
    save_path = os.path.join(
        '/home/jusun/dever120/NCVX-Neural-Structural-Optimization/results',
        f'{wandb.run.id}',
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"The directory {save_path} was created.")
    else:
        print(f"The directory {save_path} already exists.")

    # Write the configuration
    config_filepath = os.path.join(save_path, 'config.txt')
    with open(config_filepath, 'w') as f:
        f.write(f'model size = {model_size}; structure size = {structure_size}')

    # Get the device to be used
    device = utils.get_devices()
    num_trials = 1
    maxit = 1500
    max_iterations = 200

    # Set up the problem names
    # Last element is to include symmetry
    if structure_size == 'medium':
        problem_config = [
            # # Medium Size Problems
            # ("mbb_beam_96x32_0.5", False, 1, 50, False),
            # ("cantilever_beam_full_96x32_0.4", False, 1, 50, False),
            # ("michell_centered_top_64x128_0.12", True, 1, 50, False),
            # ("l_shape_0.4_128x128_0.3", False, 1, 50, False),
            # ("cantilever_beam_two_point_96x96_0.4", False, 1, 50, True),
            ("anchored_suspended_bridge_128x128_0.1", True, 1, 50, False),
        ]
    elif structure_size == 'large':
        problem_config = [
            # Large Size Problems
            # ("l_shape_0.4_192x192_0.25", False, 1, 50, False),
            # ("mbb_beam_384x128_0.5", False, 1, 50, False),
            # ("cantilever_beam_full_384x128_0.4", False, 1, 50, False),
            ("anchored_suspended_bridge_192x192_0.0875", True, 1, 50, False),
        ]

    # PyGranso function
    comb_fn = train.volume_constrained_structural_optimization_function

    # Build the problems for pygranso and google
    PYGRANSO_PROBLEMS_BY_NAME = problems.build_problems_by_name(device=device)

    # For running this we only want one trial
    # with maximum iterations 1000
    # structure_outputs = []
    for (
        problem_name,
        requires_flip,
        total_frames,
        cax_size,
        include_symmetry,
    ) in problem_config:
        print(f"Building structure: {problem_name}")
        problem = PYGRANSO_PROBLEMS_BY_NAME.get(problem_name)

        # Get volume assignment
        args = topo_api.specified_task(problem, device=device)
        volume = args["volfrac"]

        nely = int(args["nely"])
        nelx = int(args["nelx"])
        mask = (torch.broadcast_to(args["mask"], (nely, nelx)) > 0).cpu().numpy()

        # Build the structure with pygranso
        outputs = train.train_pygranso(
            problem=problem,
            device=device,
            pygranso_combined_function=comb_fn,
            requires_flip=requires_flip,
            total_frames=total_frames,
            cnn_kwargs=cnn_kwargs,
            num_trials=num_trials,
            maxit=maxit,
            include_symmetry=include_symmetry,
        )

        # Build the outputs
        pygranso_outputs = build_outputs(
            problem_name=problem_name,
            outputs=outputs,
            mask=mask,
            volume=volume,
            requires_flip=requires_flip,
        )

        # Add TOuNN to the pipeline
        tounn_outputs = tounn_train_and_outputs(problem, requires_flip)

        # Build google results - lets use our problem library
        # so we can also have custom structures not in the
        # google code
        google_problem = problem

        # Set to numpy for google framework
        google_problem.normals = google_problem.normals.cpu().numpy()
        google_problem.forces = google_problem.forces.cpu().numpy()

        if not isinstance(google_problem.mask, int):
            google_problem.mask = google_problem.mask.cpu().numpy()

        google_problem.mirror_left = True
        google_problem.mirror_right = False

        ds = train_all(google_problem, max_iterations)

        # Get google outputs
        benchmark_outputs = build_google_outputs(
            problem_name=problem_name,
            iterations=max_iterations,
            ds=ds,
            mask=mask,
            volume=volume,
            requires_flip=requires_flip,
        )

        # For each output lets save it
        # Save PyGranso Results
        pygranso_filepath = os.path.join(
            save_path, f'{problem_name}-pygranso-cnn.pickle'
        )
        with open(pygranso_filepath, 'wb') as handle:
            pickle.dump(pygranso_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        tounn_filepath = os.path.join(save_path, f'{problem_name}-tounn.pickle')
        with open(tounn_filepath, 'wb') as handle:
            pickle.dump(tounn_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        google_filepath = os.path.join(save_path, f'{problem_name}-google.pickle')
        with open(google_filepath, 'wb') as handle:
            pickle.dump(benchmark_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Run completed! 🎉')


@cli.command('run-multi-material-pipeline')
def run_multi_material_pipeline():
    """
    Function to run the multi-material pipeline
    """
    device = torch.device('cpu')
    maxit = 5

    # For testing we will run two experimentation trackers
    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Topology-Optimization'

    # Enable wandb
    wandb.login(key=API_KEY)

    # Initalize wandb
    # TODO: Save training and validation curves per fold
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=['ntopco-mmto-task'],
    )

    # Will create directories for saving models
    save_path = os.path.join(
        '/home/jusun/dever120/NCVX-Neural-Structural-Optimization/results',
        f'{wandb.run.id}',
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"The directory {save_path} was created.")
    else:
        print(f"The directory {save_path} already exists.")

    # Problem specifications
    nelx = 64
    nely = 32
    combined_frac = 0.6
    e_materials = torch.tensor([3.0, 2.0, 1.0], dtype=torch.double)
    material_density_weight = torch.tensor([1.0, 0.7, 0.4])

    args = topo_api.multi_material_tip_cantilever_task(
        nelx=nelx,
        nely=nely,
        e_materials=e_materials,
        material_density_weight=material_density_weight,
        combined_frac=combined_frac,
    )

    # Set penal to 1.0
    args['penal'] = 1.0
    args['forces'] = args['forces'].ravel()

    # Create the stiffness matrix
    ke = topo_physics.get_stiffness_matrix(
        young=args['young'],
        poisson=args['poisson'],
        device=device,
    ).double()

    # DIP Setup
    conv_filters = (256, 128, 64, 32)
    cnn_kwargs = {
        'latent_size': 128,
        'dense_channels': 96,
        'kernel_size': (5, 5),
        'conv_filters': conv_filters,
    }

    # Trials and seeds
    seeds = [0]
    for seed in seeds:
        # Intialize random seed
        utils.build_random_seed(seed)

        model = models.MultiMaterialCNNModel(args, **cnn_kwargs).to(
            device=device, dtype=torch.double
        )

        # Calculate the initial compliance
        model.eval()
        with torch.no_grad():
            initial_compliance, x_phys, _ = (
                topo_physics.calculate_multi_material_compliance(
                    model, ke, args, device, torch.double
                )
            )

        # Detach calculation and use it for scaling in PyGranso
        initial_compliance = (
            torch.ceil(initial_compliance.to(torch.float64).detach()) + 1.0
        )

        # Train PyGranso MMTO - First Stage
        # Setup the combined function for PyGranso
        comb_fn = lambda model: train.multi_material_constraint_function(  # noqa
            model,
            initial_compliance,
            ke,
            args,
            add_constraints=False,
            device=device,
            dtype=torch.double,
        )

        train.train_pygranso_mmto(
            model=model, comb_fn=comb_fn, maxit=maxit, device=device
        )

        # Train PyGranso MMTO - Second Stage
        comb_fn = lambda model: train.multi_material_constraint_function(  # noqa
            model,
            initial_compliance,
            ke,
            args,
            add_constraints=True,
            device=device,
            dtype=torch.double,
        )

        train.train_pygranso_mmto(
            model=model, comb_fn=comb_fn, maxit=maxit, device=device
        )

        # Get the final design
        compliance, final_design, _ = topo_physics.calculate_multi_material_compliance(
            model, ke, args, device, torch.double
        )
        final_design = final_design.detach().numpy()

        # Compute mass constraint
        ntopco_mass_constraint = calculate_mass_constraint(
            design=final_design,
            nelx=nelx,
            nely=nely,
            material_density_weight=material_density_weight,
            combined_frac=combined_frac,
        )

        # TODO: Extract all of the relevant information
        ntopco_outputs = {
            'final_design': final_design,
            'compliance': compliance,
            'mass_constraint': ntopco_mass_constraint,
        }

        # Compute the final design and save to experiments
        ntopco_filepath = os.path.join(save_path, f'ntopco-{seed}.pickle')
        with open(ntopco_filepath, 'wb') as handle:
            pickle.dump(ntopco_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Run MM-TOuNN Pipeline
        topOpt, mmtounn_final_design = mmtounn_train_and_outputs(
            nelx=nelx,
            nely=nely,
            e_materials=e_materials,
            material_density_weight=material_density_weight,
            combined_frac=combined_frac,
            seed=seed,
        )

        # Final compliance
        mmtounn_compliance = topOpt.convergenceHistory[-1][-1]

        # Compute mass constraint
        mmtounn_mass_constraint = calculate_mass_constraint(
            design=mmtounn_final_design,
            nelx=nelx,
            nely=nely,
            material_density_weight=material_density_weight,
            combined_frac=combined_frac,
        )

        mmtounn_outputs = {
            'final_design': mmtounn_final_design,
            'compliance': mmtounn_compliance,
            'mass_constraint': mmtounn_mass_constraint,
        }

        # Compute the final design and save to experiments
        mmtounn_filepath = os.path.join(save_path, f'mmtounn-{seed}.pickle')
        with open(mmtounn_filepath, 'wb') as handle:
            pickle.dump(mmtounn_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)


@cli.command('run-multi-structure-pygranso-pipeline')
def run_multi_structure_pygranso_pipeline():
    """
    Task that will build out multiple structures and compare
    performance against known benchmarks.
    """
    # Model size
    model_size = 'medium'

    # CNN parameters
    cnn_features = (256, 128, 64, 32, 16)

    # Configurations
    configs = {
        'tiny': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 6 for features in cnn_features),
        },
        'xsmall': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 5 for features in cnn_features),
        },
        'small': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 4 for features in cnn_features),
        },
        'medium': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 3 for features in cnn_features),
        },
        'large': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 2 for features in cnn_features),
        },
        # x-large has been our original architecture
        'xlarge': {
            'latent_size': 96,
            'dense_channels': 32,
            'conv_filters': tuple(features // 1 for features in cnn_features),
        },
    }

    # CNN kwargs
    cnn_kwargs = configs[model_size]

    # Set seed
    models.set_seed(0)  # Model seed is set here but results are changing?

    # For testing we will run two experimentation trackers
    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Topology-Optimization'

    # Enable wandb
    wandb.login(key=API_KEY)

    # Initalize wandb
    # TODO: Save training and validation curves per fold
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=['topology-optimization-pg-seed-task', model_size],
        config=cnn_kwargs,
    )

    # Get the device to be used
    device = utils.get_devices()
    num_trials = 5
    maxit = 1500

    # Set up the problem names
    problem_config = [
        # Medium Size Problems
        ("mbb_beam_96x32_0.5", True, 1, 50),
        ("cantilever_beam_full_96x32_0.4", True, 1, 50),
        ("michell_centered_top_64x128_0.12", True, 1, 50),
        ("l_shape_0.4_128x128_0.3", True, 1, 50),
        ("suspended_bridge_192x192_0.0875", True, 1, 50),
    ]

    # renaming
    name_mapping = {
        # Medium Size Problems
        'mbb_beam_96x32_0.5': 'MBB \n Beam',
        'cantilever_beam_full_96x32_0.4': 'Cantilever \n Beam',
        'michell_centered_top_64x128_0.12': 'Michell \n Top',
        'l_shape_0.4_128x128_0.3': 'L-Shape \n 0.4',
        "suspended_bridge_128x128_0.1": "Anchored Suspended \n Bridge",
    }

    # PyGranso function
    comb_fn = train.volume_constrained_structural_optimization_function

    # Build the problems for pygranso and google
    PYGRANSO_PROBLEMS_BY_NAME = problems.build_problems_by_name(device=device)

    # For running this we only want one trial
    # with maximum iterations 1000
    structure_outputs = []
    for problem_name, requires_flip, total_frames, cax_size in problem_config:
        print(f"Building structure: {problem_name}")
        problem = PYGRANSO_PROBLEMS_BY_NAME.get(problem_name)

        # Build the structure with pygranso
        outputs = train.train_pygranso(
            problem=problem,
            device=device,
            pygranso_combined_function=comb_fn,
            requires_flip=requires_flip,
            total_frames=total_frames,
            cnn_kwargs=cnn_kwargs,
            neptune_logging=None,
            num_trials=num_trials,
            maxit=maxit,
        )
        structure_outputs.append(outputs)

    # Plot the different seeds
    fig, axes = plt.subplots(len(problem_config), num_trials, figsize=(15, 6))
    fig.subplots_adjust(wspace=0, hspace=0)

    # Get the updated structure names
    structures = list(name_mapping.values())

    # iterate over the structures and plot different seeds
    for index, structure_name in enumerate(structures):
        # Get the designs
        designs = structure_outputs[index]['designs']

        # Get the losses
        losses = structure_outputs[index]['losses']
        losses = pd.DataFrame(losses).ffill()
        losses = losses.iloc[-1, :]
        print(f'All losses {losses}')
        print('\n')
        print(f'Median loss for {structure_name}; {np.median(losses)}')

        for trial in range(num_trials):
            ax = axes[index, trial]
            design = designs[trial, :, :]
            design = np.hstack((design[:, ::-1], design))
            im = ax.imshow(design, cmap='jet', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add the ylabel (structure name)
            if trial == 0:
                ax.set_ylabel(f'{structure_name}', fontsize=14, weight='bold')

    fig.colorbar(im, ax=axes.ravel().tolist())

    # Also, save fig
    fig.savefig(
        f'/home/jusun/dever120/NCVX-Neural-Structural-Optimization/results/{model_size}-pygranso-seed-results.png',  # noqa
        bbox_inches='tight',
    )

    # Save figure to weights and biases
    wandb.log({'plot': wandb.Image(fig)})


if __name__ == "__main__":
    cli()
