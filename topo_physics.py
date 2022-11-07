# third party
import numpy as np
import scipy
import torch

# first party
import utils


# Calculate the young modulus
def young_modulus(x, e_0, e_min, p=3):
    """
    Function that calculates the young modulus
    """
    e_0 = torch.tensor(e_0)
    e_min = torch.tensor(e_min)
    p = torch.tensor(p)

    return e_min + x**p * (e_0 - e_min)


# Define the physical density with torch
def physical_density(x, args, volume_constraint=True, filtering=False):
    """
    Function that calculates the physical density
    """
    shape = (args["nely"], args["nelx"])
    arg_mask = args["mask"]
    size_x = len(x.flatten())

    # In the code they do a reshape but this would not be necessary
    # if this assertion is broken
    assert x.shape == shape or x.ndim == 1

    if volume_constraint:
        if len(arg_mask.flatten()) == 1:
            x = utils.sigmoid_with_constrained_mean(x, args["volfrac"])

        else:
            mask = torch.broadcast_to(arg_mask, shape) > 0
            x = utils.sigmoid_with_constrained_mean(x[mask], args["volfrac"])
            flat_nonzero_mask = torch.nonzero(mask.ravel(), as_tuple=True)[0]
            x = utils.torch_scatter1d(x, flat_nonzero_mask, size_x)
            x = x.reshape(shape)

    else:
        x = x * args["mask"]

    if filtering:
        # TODO: there was a 2D gaussian filter here
        # but I removed it because I do not know
        # how it will react with pytorch
        pass

    return x


# Calculate the forces
def calculate_forces(x_phys, args):
    """
    Function that just gathers the current forces from the
    problem
    """
    # For now this is all that is happening in the code although
    # we should be aware of this if this could be an issue with
    # harder problems. This eliminates the gravitational force
    # on the object
    return args["forces"]


# Build a stiffness matrix
def get_stiffness_matrix(young: float, poisson: float) -> np.array:
    """
    Function to build the elements of the stiffness matrix
    """
    # Build the element stiffness matrix
    # What are e & nu?
    e, nu = young, poisson

    # I want to understand where these elements come from
    k = torch.tensor(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )

    # We want to shuffle the k matrix with different indices
    zero_row_shuffle = torch.arange(len(k))
    first_row_shuffle = torch.tensor([1, 0, 7, 6, 5, 4, 3, 2])
    second_row_shuffle = torch.tensor([2, 7, 0, 5, 6, 3, 4, 1])
    third_row_shuffle = torch.tensor([3, 6, 5, 0, 7, 2, 1, 4])
    fourth_row_shuffle = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])
    fifth_row_shuffle = torch.tensor([5, 4, 3, 2, 1, 0, 7, 6])
    sixth_row_shuffle = torch.tensor([6, 3, 4, 1, 2, 7, 0, 5])
    seventh_row_shuffle = torch.tensor([7, 2, 1, 4, 3, 6, 5, 0])

    # Create shuffled array
    shuffled_array = torch.stack(
        [
            k[zero_row_shuffle],
            k[first_row_shuffle],
            k[second_row_shuffle],
            k[third_row_shuffle],
            k[fourth_row_shuffle],
            k[fifth_row_shuffle],
            k[sixth_row_shuffle],
            k[seventh_row_shuffle],
        ]
    )

    return e / (1 - nu**2) * shuffled_array


# Compliance
def compliance(x_phys, u, ke, args, *, penal=3, e_min=1e-9, e_0=1, base="Google"):
    """
    Calculate the compliance objective.

    NOTE: For our implementation both x_phys and u will require_grad
    and will both be torch tensors.
    """
    # nely, nelx = x_phys.shape
    # ely, elx = np.meshgrid(range(nely), range(nelx))

    # # Nodes
    # n1 = (nely + 1) * (elx + 0) + (ely + 0)
    # n2 = (nely + 1) * (elx + 1) + (ely + 0)
    # n3 = (nely + 1) * (elx + 1) + (ely + 1)
    # n4 = (nely + 1) * (elx + 0) + (ely + 1)
    # all_ixs = np.array(
    #     [
    #         2 * n1,
    #         2 * n1 + 1,
    #         2 * n2,
    #         2 * n2 + 1,
    #         2 * n3,
    #         2 * n3 + 1,
    #         2 * n4,
    #         2 * n4 + 1,
    #     ]  # noqa
    # )
    # all_ixs = torch.from_numpy(all_ixs)

    nely, nelx = args["nely"], args["nelx"]

    # Compute the torch based meshgrid
    ely, elx = torch.meshgrid(torch.arange(nely), torch.arange(nelx), indexing="xy")

    # Calculate nodes - reflects the MATLAB code
    if base == "MATLAB":
        n1 = (nely + 1) * (elx + 0) + (ely + 1)
        n2 = (nely + 1) * (elx + 1) + (ely + 1)
        n3 = (nely + 1) * (elx + 1) + (ely + 0)
        n4 = (nely + 1) * (elx + 0) + (ely + 0)

    elif base == "Google":
        # Google implementation
        n1 = (nely + 1) * (elx + 0) + (ely + 0)
        n2 = (nely + 1) * (elx + 1) + (ely + 0)
        n3 = (nely + 1) * (elx + 1) + (ely + 1)
        n4 = (nely + 1) * (elx + 0) + (ely + 1)

    else:
        raise ValueError("Only options are MATLAB and Google!")

    # The shape of this matrix results in
    # (8, nelx, nely)
    all_ixs = torch.stack(
        [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]
    )

    # The selected u and now needs to be multiplied by K
    u_selected = u[all_ixs].squeeze()

    # Set ke to have double
    ke = ke.double()

    # Run the compliance calculation
    ke_u = torch.einsum("ij,jkl->ikl", ke, u_selected)
    ce = torch.einsum("ijk,ijk->jk", u_selected, ke_u)
    young_x_phys = young_modulus(x_phys, e_0, e_min, p=penal)

    return young_x_phys * ce.T, u_selected, ke_u


def get_k_data(stiffness, ke, args, base="MATLAB"):
    """
    Function that is the pytorch version of get_K
    from the other repository
    """
    if not torch.is_tensor(ke):
        ke = torch.from_numpy(ke)

    # Get the dimensions
    nely, nelx = args["nely"], args["nelx"]

    edof, x_list, y_list = build_nodes_data(args, base=base)

    kd = stiffness.T.reshape(nely * nelx, 1, 1)
    value_list = (kd * ke.tile(kd.shape)).flatten()

    return value_list, y_list, x_list


def displace(x_phys, ke, forces, freedofs, fixdofs, *, penal=3, e_min=1e-9, e_0=1):
    """
    Function that displaces the load x using finite element techniques.
    """
    stiffness = young_modulus(x_phys, e_0, e_min, p=penal)

    # Get the K values
    k_entries, k_ylist, k_xlist = get_k(stiffness, ke)

    index_map, keep, indices = utils._get_dof_indices(
        freedofs, fixdofs, k_ylist, k_xlist
    )

    # Reduced forces
    freedofs_forces = forces[freedofs].double()

    # K matrix based on the size of forces[freedofs]
    K = (
        torch.sparse_coo_tensor(
            indices, k_entries[keep], (len(freedofs_forces),) * 2
        ).to_dense()
    ).double()
    K = (K + K.transpose(1, 0)) / 2.0

    # Compute the non-zero u values
    # Get the choleskly factorization
    k_cholesky = torch.linalg.cholesky(K)

    # Solve for u nonzero
    u_nonzero = torch.cholesky_solve(
        freedofs_forces.reshape(len(freedofs_forces), 1),
        k_cholesky,
    ).flatten()
    u_values = torch.cat((u_nonzero, torch.zeros(len(fixdofs))))

    return u_values[index_map], K


def sparse_displace(
    x_phys,
    ke,
    args,
    forces,
    freedofs,
    fixdofs,
    *,
    penal=3,
    e_min=1e-9,
    e_0=1,
    base="Google",
):
    """
    Function that displaces the load x using finite element techniques.
    """
    stiffness = young_modulus(x_phys, e_0, e_min, p=penal)

    # Get the K values
    k_entries, k_ylist, k_xlist = get_k_data(stiffness, ke, args, base=base)

    # Get the full indices
    full_indices = torch.stack([k_ylist, k_xlist])

    index_map, keep, indices = utils._get_dof_indices(
        freedofs, fixdofs, k_ylist, k_xlist
    )

    # Reduced forces
    freedofs_forces = forces[freedofs].double()
    size = freedofs_forces.numpy().size

    # Require gradient on the forces
    freedofs_forces = freedofs_forces.requires_grad_()

    # Calculate u_nonzero
    keep_k_entries = k_entries[keep]

    # Build the sparse matrix
    K = torch.sparse_coo_tensor(indices, keep_k_entries, [size, size])
    K = (K + K.transpose(1, 0)) / 2.0

    # Symmetric indices
    keep_k_entries = K.coalesce().values()
    indices = K.coalesce().indices()

    # Compute the u_matrix values
    u_nonzero = utils.solve_coo(keep_k_entries, indices, freedofs_forces, sym_pos=False)
    u_values = torch.cat((u_nonzero, torch.zeros(len(fixdofs))))

    return u_values[index_map], None


def build_K_matrix(x_phys, args, base="MATLAB"):
    """
    Function to build the full K matrix
    """
    # Build the stiffness using young's modulus
    stiffness = young_modulus(
        x_phys, e_0=args["young"], e_min=args["young_min"], p=args["penal"]
    )

    # Build KE
    ke = get_stiffness_matrix(young=args["young"], poisson=args["poisson"])

    # Get the K indices and values
    k_entries, k_ylist, k_xlist = get_k_data(
        stiffness=stiffness, ke=ke, args=args, base=base
    )

    # Create full indices
    if base == "MATLAB":
        indices = torch.stack([k_ylist, k_xlist])
        k_entries = k_entries

        # Create the K matrix
        K = (torch.sparse_coo_tensor(indices, k_entries)).to_dense().double()
        K = (K + K.transpose(1, 0)) / 2.0

    elif base == "Google":
        freedofs = args["freedofs"].numpy()
        fixdofs = args["fixdofs"].numpy()
        free_forces = args["forces"][freedofs].numpy()

        # Get the indices
        index_map, keep, indices = utils._get_dof_indices(
            freedofs, fixdofs, k_ylist, k_xlist
        )
        size = free_forces.size

        # Calculate the K matrix for the google based code
        k_entires = k_entries[keep].numpy()
        indices = indices.numpy()

        # Compute K
        K = (
            scipy.sparse.coo_matrix((k_entries[keep], indices), shape=(size,) * 2)
            .tocsc()
            .todense()
        )
        K = (K + K.T) / 2.0

    else:
        raise ValueError("Only methods MATLAB and Google are valid!")

    return K


def build_nodes_data(args, base="MATLAB"):
    """
    Build the nodes matrix from both the MATLAB and Google
    code. Turns out the code from the two different methods is
    different but we want to be able to test that we have
    the right implementation for each method. These same nodes will
    also be fed into the compliance objective.
    """
    nely, nelx = args["nely"], args["nelx"]

    # Compute the torch based meshgrid
    ely, elx = torch.meshgrid(torch.arange(nely), torch.arange(nelx), indexing="xy")
    ely, elx = ely.reshape(-1, 1), elx.reshape(-1, 1)

    # Calculate nodes - reflects the MATLAB code
    if base == "MATLAB":
        n1 = (nely + 1) * (elx + 0) + (ely + 1)
        n2 = (nely + 1) * (elx + 1) + (ely + 1)
        n3 = (nely + 1) * (elx + 1) + (ely + 0)
        n4 = (nely + 1) * (elx + 0) + (ely + 0)

    elif base == "Google":
        # Google implementation
        n1 = (nely + 1) * (elx + 0) + (ely + 0)
        n2 = (nely + 1) * (elx + 1) + (ely + 0)
        n3 = (nely + 1) * (elx + 1) + (ely + 1)
        n4 = (nely + 1) * (elx + 0) + (ely + 1)

    else:
        raise ValueError("Only options are MATLAB and Google!")

    # The shape of this matrix results in
    # (8, nelx, nely)
    edof = torch.stack(
        [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]
    )

    # Calculate the transpose & get the first
    # element
    edof = edof.permute(2, 1, 0)[0]

    # Build the i & j indices for a sparse matrix
    y_list = edof.tile((8,)).flatten()
    x_list = edof.repeat_interleave(8)

    return edof, y_list, x_list


def build_nodes_data_google(args):
    """
    An exact implementation of the Google code to build
    the nodes for K and the compliance
    """
    nely, nelx = args["nely"], args["nelx"]

    # get position of the nodes of each element in the stiffness matrix
    ely, elx = np.meshgrid(range(nely), range(nelx))  # x, y coords
    ely, elx = ely.reshape(-1, 1), elx.reshape(-1, 1)

    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)
    edof = np.array(
        [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]
    )
    edof = edof.T[0]

    return edof


def get_stiffness_matrix_google(young, poisson):
    """
    Function from the google code to compute the stiffness
    matrix
    """
    # Element stiffness matrix
    e, nu = young, poisson
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )
    return (
        e
        / (1 - nu**2)
        * np.array(
            [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ]
        )
    )
