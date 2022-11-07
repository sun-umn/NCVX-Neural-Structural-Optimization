# stdlib
import os
from pathlib import Path

# third part
import numpy as np
import scipy
import torch
from torch.autograd import gradcheck

# first party
import problems
import topo_api
import topo_physics
import utils


def test_matlab_edofmat():
    """
    Test that the edofMat (based on the MATLAB code)
    is equal to the nodes that we build for K and the compliance
    """
    # Setup fixture path
    fixture_path = str(Path(__file__).parent / "fixtures")

    # Set up dummy args for the test
    args = {}
    args["nely"] = 3
    args["nelx"] = 4

    # Matlab data
    matlab_edofMat = scipy.io.loadmat(os.path.join(fixture_path, "edofMat.mat"))  # noqa
    matlab_edofMat = matlab_edofMat["edofMat"]  # noqa

    # Get the output from topo_physics
    edof, _, _ = topo_physics.build_nodes_data(args, base="MATLAB")

    # Convert into numpy array
    # For the thist test we need to add 1 to our outputs
    # because MATLAB is one indexed
    edof = edof.numpy() + 1

    # Assert the frames are equal
    np.testing.assert_allclose(matlab_edofMat, edof)


def test_google_edofmat():
    """
    Test that the edofMat (based on the Google code)
    is equal to the nodes that we build for K and the compliance
    when the base = 'Google'
    """
    # Set up dummy args for the test
    args = {}
    args["nely"] = 20
    args["nelx"] = 60

    # Google edof
    google_edof = topo_physics.build_nodes_data_google(args)

    # Get the output from topo_physics
    edof, _, _ = topo_physics.build_nodes_data(args, base="Google")

    # Assert the frames are equal
    np.testing.assert_allclose(google_edof, edof)


def test_matlab_k_output():
    """
    Test that the K matrix (from the MATLAB output) is equal to the
    full K matrix that we build with the correct MATLAB outputs nodes
    """
    # Setup fixture path
    fixture_path = str(Path(__file__).parent / "fixtures")

    # Set up dummy args for the test
    args = {}
    args["nely"] = 3
    args["nelx"] = 4

    # Set penal = 1.0 like in the MATLAB code
    args["penal"] = 1.0
    args["young"] = 1.0
    args["young_min"] = 1e-9
    args["poisson"] = 0.3

    # Matlab data
    matlab_K = scipy.io.loadmat(os.path.join(fixture_path, "K.mat"))  # noqa
    matlab_K = matlab_K["K"].todense()  # noqa

    # Setup
    nely, nelx = args["nely"], args["nelx"]
    x_phys = torch.ones(nely, nelx) * 0.5

    K = topo_physics.build_K_matrix(x_phys, args, base="MATLAB")  # noqa
    K = K.numpy()  # noqa

    # Assert the shapes are equal
    assert matlab_K.shape == K.shape

    # Assert all the values are very close
    np.testing.assert_allclose(matlab_K, K)


def test_matlab_k_freedofs_output():
    """
    Test that the K matrix (from the MATLAB output) with the freedofs
    is equal to what we can also produce
    """
    # Setup fixture path
    fixture_path = str(Path(__file__).parent / "fixtures")

    # Set up dummy args for the test
    # We will need to set up the real problem
    problem = problems.mbb_beam(height=3, width=4)
    problem.name = "mbb_beam"

    # Get the problem args
    args = topo_api.specified_task(problem)
    args["penal"] = 1.0

    # Matlab data
    matlab_K_freedofs = scipy.io.loadmat(  # noqa
        os.path.join(fixture_path, "K_freedofs.mat")
    )  # noqa
    matlab_K_freedofs = matlab_K_freedofs["K_freedofs"].todense()  # noqa

    # Setup
    nely, nelx = args["nely"], args["nelx"]
    x_phys = torch.ones(nely, nelx) * 0.5
    freedofs = args["freedofs"].numpy()

    K_freedofs = topo_physics.build_K_matrix(x_phys, args, base="MATLAB")  # noqa
    K_freedofs = scipy.sparse.csr_matrix(K_freedofs.numpy())  # noqa
    K_freedofs = K_freedofs[freedofs, :][:, freedofs].todense()  # noqa

    # Assert the shapes are equal
    assert matlab_K_freedofs.shape == K_freedofs.shape

    # Assert all the values are very close
    np.testing.assert_almost_equal(matlab_K_freedofs, K_freedofs, decimal=3)


def test_google_k_output():
    """
    Test that we also can correcly compute the K matrix from
    the Google based code which is different from the code that
    MATLAB provides
    """
    # Setup fixture path
    fixture_path = str(Path(__file__).parent / "fixtures")

    # We will need to set up the real problem
    problem = problems.mbb_beam(height=3, width=4)
    problem.name = "mbb_beam"

    # Get the problem args
    args = topo_api.specified_task(problem)
    args["penal"] = 1.0

    # google data
    google_K = np.load(os.path.join(fixture_path, "sparse_K.npy"))  # noqa

    # Setup
    nely, nelx = args["nely"], args["nelx"]
    x_phys = torch.ones(nely, nelx) * 0.5

    K = topo_physics.build_K_matrix(x_phys, args, base="Google")  # noqa
    K = np.round(K, 3)  # noqa

    # Assert the shapes are equal
    assert google_K.shape == K.shape

    # Assert all the values are very close
    np.testing.assert_allclose(google_K, K)


def test_goggle_not_equal_matlab():
    """
    There seem to be some discrepancies between the two methods.
    We have saved both data arrays so we will test that they
    are NOT equal
    """
    # Setup fixture path
    fixture_path = str(Path(__file__).parent / "fixtures")

    # google data
    google_K = np.load(os.path.join(fixture_path, "sparse_K.npy"))  # noqa

    # Matlab data
    matlab_K_freedofs = scipy.io.loadmat(  # noqa
        os.path.join(fixture_path, "K_freedofs.mat")
    )  # noqa
    matlab_K_freedofs = matlab_K_freedofs["K_freedofs"].todense()  # noqa

    assert google_K.shape == matlab_K_freedofs.shape

    # Assert values are not equal
    assert np.any(google_K != matlab_K_freedofs)


def test_stiffness_matrix_is_equal():
    """
    Testing that the KE is equivalent in our code
    as well as the google code
    """
    ke = topo_physics.get_stiffness_matrix(young=1.0, poisson=0.3)
    ke_google = topo_physics.get_stiffness_matrix_google(young=1.0, poisson=0.3)

    np.testing.assert_allclose(ke, ke_google)


def test_sparse_solver_grad_check():
    """
    Test that checks if our gradient for the sparse
    linear solver is correct
    """
    A = torch.randn(3, 3, requires_grad=True)  # noqa

    # Since we are returning a vector - we noticed that in order
    # for this to work the results will need to be symmetric
    A = (A + A.transpose(1, 0)) / 2.0  # noqa
    b = torch.randn(3, requires_grad=True)

    # Create the indices for the sparse matrix
    i = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    j = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    indices = torch.stack([i, j])

    # Build the solver
    solver = utils.SparseSolver.apply

    # Check the gradient
    assert gradcheck(solver, [A.flatten().double(), indices, b.double(), False])
