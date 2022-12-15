# stdlib
import os
from pathlib import Path

# third party
import numpy as np
import scipy
import sksparse.cholmod
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


# Specific tests for all of the MBB Beam problems
def test_mbb_beam_384_64_40():
    """
    We will fully test the values between the MATLAB version
    and our version - what we found was that the values were
    not matching up 100%
    """
    # Setup problem
    problem = problems.mbb_beam(
        height=64,
        width=384,
        density=0.4,
        device=utils.DEFAULT_DEVICE,
    )
    args = topo_api.specified_task(problem, device=utils.DEFAULT_DEVICE)

    # Test inputs to the sparse matrix
    fixture_path = str(Path(__file__).parent / "fixtures")
    iK = np.loadtxt(os.path.join(fixture_path, "mbb_384_64_40_iK.mat"))  # noqa
    jK = np.loadtxt(os.path.join(fixture_path, "mbb_384_64_40_jK.mat"))  # noqa
    sK = np.loadtxt(os.path.join(fixture_path, "mbb_384_64_40_sK.mat"))  # noqa
    matlab_K_free = np.loadtxt(  # noqa
        os.path.join(fixture_path, "mbb_384_64_40_K_free.mat")
    )  # noqa

    # Get ke
    ke = topo_physics.get_stiffness_matrix(
        young=args["young"],
        poisson=args["poisson"],
        device=utils.DEFAULT_DEVICE,
    )

    # Set up x_phys
    nely, nelx = args["nely"], args["nelx"]
    x_phys = torch.ones(nely, nelx) * 0.4

    # Free and fixed degress of freedom
    freedofs = args["freedofs"]
    fixdofs = args["fixdofs"]

    # Reduced forces
    forces = topo_physics.calculate_forces(x_phys, args)
    freedofs_forces = (
        forces[freedofs]
        .double()
        .to(device=utils.DEFAULT_DEVICE, dtype=utils.DEFAULT_DTYPE)
    )
    size = freedofs_forces.cpu().numpy().size

    # setup stiffness
    stiffness = topo_physics.young_modulus(
        x_phys, e_0=args["young"], e_min=args["young_min"], p=args["penal"]
    )

    # Get K Data
    k_entries, k_ylist, k_xlist = topo_physics.get_k_data(
        stiffness, ke, args, base="MATLAB"
    )

    # Assert all entities are equal
    np.testing.assert_allclose(k_ylist, jK - 1)
    np.testing.assert_allclose(k_xlist, iK - 1)
    np.testing.assert_almost_equal(k_entries, sK, decimal=4)

    # Get the indices for the free degress of freedom
    index_map, keep, indices = utils._get_dof_indices(
        freedofs, fixdofs, k_ylist, k_xlist, k_entries
    )

    # Next create the sparse matrix - let's test the full and
    # freedof versions
    keep_k_entries = k_entries[keep]
    K = (
        torch.sparse_coo_tensor(indices, keep_k_entries, [size, size])  # noqa
        .double()
        .coalesce()
    )  # noqa
    K = K.to_dense().to_sparse_coo()  # noqa

    # Get indices and values to check
    indices = K.indices().t().int()
    values = K.values()

    # Testing matrix that goes into solver
    np.testing.assert_allclose(indices[:, 0], matlab_K_free[:, 1] - 1)
    np.testing.assert_allclose(indices[:, 1], matlab_K_free[:, 0] - 1)
    np.testing.assert_almost_equal(values, matlab_K_free[:, 2], decimal=4)

    # Now test the u_values
    row_mat = matlab_K_free[:, 0] - 1
    col_mat = matlab_K_free[:, 1] - 1
    a_entries_mat = np.round(matlab_K_free[:, 2], 7)
    b = freedofs_forces.detach().cpu().numpy()

    a_matlab = scipy.sparse.csc_matrix(
        (a_entries_mat, (row_mat, col_mat)), shape=(b.size,) * 2
    ).astype(np.float64)

    # Solver for matlab data
    solver_mat = sksparse.cholmod.cholesky(a_matlab).solve_A

    # Test the u values that we are creating
    row = K.indices().t()[:, 1].cpu().numpy()
    col = K.indices().t()[:, 0].cpu().numpy()

    # We prove above that they are equal if it gets to this
    # stage
    a_entries = np.round(values.cpu().numpy(), 7)
    b = freedofs_forces.detach().cpu().numpy()

    a = scipy.sparse.csc_matrix((a_entries, (row, col)), shape=(b.size,) * 2).astype(
        np.float64
    )

    # Our values
    solver = sksparse.cholmod.cholesky(a).solve_A

    # Test everything
    np.testing.assert_allclose(row_mat, row)
    np.testing.assert_allclose(col_mat, col)
    np.testing.assert_almost_equal(a_entries_mat, a_entries, decimal=5)
    assert (a != a_matlab).nnz == 0

    # Check the u values
    u_matlab = solver_mat(b)
    u_ = solver(b)

    # Assert that we have the same displacement values
    np.testing.assert_allclose(u_matlab, u_)

    # Finally assert that we have the same compliance values
    kwargs = dict(
        penal=args["penal"],
        e_min=args["young_min"],
        e_0=args["young"],
        base="MATLAB",
        device=utils.DEFAULT_DEVICE,
        dtype=utils.DEFAULT_DTYPE,
    )

    # Fixed zeros
    fixdofs_zeros = torch.zeros(len(fixdofs)).to(
        device=utils.DEFAULT_DEVICE, dtype=utils.DEFAULT_DTYPE
    )

    # Updated u matlab
    u_matlab = torch.from_numpy(u_matlab)
    u_values_mat = torch.cat((u_matlab, fixdofs_zeros))
    u_values_mat = u_values_mat[index_map].to(
        device=utils.DEFAULT_DEVICE, dtype=utils.DEFAULT_DTYPE
    )

    # Calculate the compliance output
    compliance_output, _, _ = topo_physics.compliance(
        x_phys, u_values_mat, ke, args, **kwargs
    )
    compliance_output = torch.sum(compliance_output)

    u_ = torch.from_numpy(u_)
    u_values = torch.cat((u_, fixdofs_zeros))
    u_values = u_values[index_map].to(
        device=utils.DEFAULT_DEVICE, dtype=utils.DEFAULT_DTYPE
    )

    compliance_output_mat, _, _ = topo_physics.compliance(
        x_phys, u_values, ke, args, **kwargs
    )
    compliance_output_mat = torch.sum(compliance_output_mat)

    print(u_values_mat, u_values)

    assert compliance_output == compliance_output_mat
