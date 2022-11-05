# stdlib
import os
from pathlib import Path

# third part
import numpy as np
import scipy
import torch

# first party
import problems
import topo_api
import topo_physics


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
    args["nely"] = 3
    args["nelx"] = 4

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
