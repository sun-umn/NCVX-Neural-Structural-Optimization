import numpy as np
import torch
from str_opt_utils import utils
import scipy.sparse

# Calculate the young modulus
def young_modulus(x, e_0, e_min, p=3):
    """
    Function that calculates the young modulus
    """
    # e_0 = e_0.clone().detach()
    # e_min = e_min.clone().detach()
    # p = p.clone().detach()

    return e_min + x ** p * (e_0 - e_min)


# Define the physical density with torch
def physical_density(x, args, volume_constraint=True, filtering=False):
    """
    Function that calculates the physical density
    """
    shape = (args["nely"], args["nelx"])
    arg_mask = args['mask']
    size_x = len(x.flatten())

    # In the code they do a reshape but this would not be necessary
    # if this assertion is broken
    assert x.shape == shape or x.ndim == 1

    if volume_constraint:
        if len(arg_mask.flatten()) == 1:
            x = utils.sigmoid_with_constrained_mean(x, args['volfrac'])

        else:
            mask = torch.broadcast_to(arg_mask, shape) > 0
            x = (
                utils
                .sigmoid_with_constrained_mean(x[mask], args["volfrac"])
            )
            flat_nonzero_mask = torch.nonzero(
                mask.ravel(), as_tuple=True
            )[0]
            x = utils.torch_scatter1d(
                x, flat_nonzero_mask, size_x
            )
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
def get_stiffness_matrix(young: float, poisson: float, device=torch.device('cpu'), dtype=torch.double) -> torch.tensor:
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
    ).to(device=device, dtype=dtype)

    return e / (1 - nu ** 2) * shuffled_array


# Compliance
def compliance(x_phys, u, ke, *, penal=3, e_min=1e-9, e_0=1,device=torch.device('cpu'), dtype=torch.double):
    """
    Calculate the compliance objective.

    NOTE: For our implementation both x_phys and u will require_grad
    and will both be torch tensors.
    """
    nely, nelx = x_phys.shape
    ely, elx = np.meshgrid(range(nely), range(nelx))

    # Nodes
    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)
    all_ixs = np.array(
        [
            2 * n1,
            2 * n1 + 1,
            2 * n2,
            2 * n2 + 1,
            2 * n3,
            2 * n3 + 1,
            2 * n4,
            2 * n4 + 1,
        ]  # noqa
    )
    all_ixs = torch.from_numpy(all_ixs)

    # The selected u and now needs to be multiplied by K
    u_selected = u[all_ixs].squeeze()

    # Set ke to have double
    ke = ke.double()

    # Run the compliance calculation
    ke_u = torch.einsum("ij,jkl->ikl", ke, u_selected)
    ce = torch.einsum("ijk,ijk->jk", u_selected, ke_u)
    young_x_phys = young_modulus(x_phys, e_0, e_min, p=penal)

    return young_x_phys * ce.T


def get_k(stiffness, ke):
    """
    Function that is the pytorch version of get_K
    from the other repository
    """
    if not torch.is_tensor(ke):
        ke = torch.from_numpy(ke)

    nely, nelx = stiffness.shape

    # Compute the torch based meshgrid
    # Buyun: indexing not allowed for torch 1.9.0
    # ely, elx = torch.meshgrid(torch.arange(nely), torch.arange(nelx), indexing="xy")
    elx,ely = torch.meshgrid(torch.arange(nelx),torch.arange(nely))

    ely, elx = ely.reshape(-1, 1), elx.reshape(-1, 1)

    # Calculate nodes
    n1 = (nely + 1) * (elx + 0) + (ely + 0)
    n2 = (nely + 1) * (elx + 1) + (ely + 0)
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * (elx + 0) + (ely + 1)

    # The shape of this matrix results in
    # (8, nelx, nely)
    edof = torch.stack(
        [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]
    )

    # Calculate the transpose & get the first
    # element
    edof = edof.permute(2, 1, 0)[0]

    # Create the x & y list
    x_list = edof.repeat_interleave(8)
    y_list = edof.tile((8,)).flatten()
    kd = stiffness.T.reshape(nely * nelx, 1, 1)
    value_list = (kd * ke.tile(kd.shape)).flatten()

    return value_list, y_list, x_list


def get_K(x_phys,ke,args,kwargs):
    # Calculate the forces
    forces = calculate_forces(x_phys=None, args=args)
    
    # Instead of calculating u_matrix let's see if we can learn it
    # Calculate the stiffness matrix
    stiffness = young_modulus(
        x_phys, e_min=args['young_min'], e_0=args['young'], p=args['penal'],
    )
    
    # # Define freedofs and fixdofs
    # freedofs = args['freedofs']
    # fixdofs = args['fixdofs']
    
    # Get the K values
    k_entries, k_ylist, k_xlist = get_k(stiffness, ke)
    k_ylist = k_ylist.to(device=kwargs['device'], dtype=kwargs['dtype'])
    k_xlist = k_xlist.to(device=kwargs['device'], dtype=kwargs['dtype'])
    
    full_indices = torch.stack([k_ylist, k_xlist])
    K = (
        torch.sparse_coo_tensor(
            full_indices, k_entries, (len(forces),) * 2
        ).to_dense()
    ).double()
    K = (K + K.transpose(1, 0)) / 2.0
    return K

def displace(x_phys, ke, forces, freedofs, fixdofs, *, penal=3, e_min=1e-9, e_0=1,device=torch.device('cpu'), dtype=torch.double):
    """
    Function that displaces the load x using finite element techniques.
    """
    stiffness = young_modulus(x_phys, e_0, e_min, p=penal)

    # Get the K values
    k_entries, k_ylist, k_xlist = get_k(stiffness, ke)
    k_ylist = k_ylist.to(device=device, dtype=dtype)
    k_xlist = k_xlist.to(device=device, dtype=dtype)

    index_map, keep, indices = utils._get_dof_indices(
        freedofs, fixdofs, k_ylist, k_xlist
    )

    # Reduced forces
    freedofs_forces = forces[freedofs.cpu().numpy()].double()

    # K matrix based on the size of forces[freedofs]
    K = (
        torch.sparse_coo_tensor(
            indices, k_entries[keep], (len(freedofs_forces),) * 2
        ).to_dense()
    ).double()
    K = (K + K.transpose(1, 0)) / 2.0

    # Compute the inverse of K
    K_inverse = torch.inverse(K)

    # Compute the non-zero u values
    u_nonzero = K_inverse @ freedofs_forces
    u_values = torch.cat((u_nonzero, torch.zeros(len(fixdofs)).to(device=device, dtype=dtype)))

    return u_values[index_map], K


def sparse_displace(x_phys, ke, forces, freedofs, fixdofs, *, penal=3, e_min=1e-9, e_0=1,device=torch.device('cpu'), dtype=torch.double):
    """
    Function that displaces the load x using finite element techniques.
    """
    stiffness = young_modulus(x_phys, e_0, e_min, p=penal)

    # Get the K values
    k_entries, k_ylist, k_xlist = get_k(stiffness, ke)
    k_ylist = k_ylist.to(device=device, dtype=dtype)
    k_xlist = k_xlist.to(device=device, dtype=dtype)

    index_map, keep, indices = utils._get_dof_indices(
        freedofs, fixdofs, k_ylist, k_xlist
    )

    # Reduced forces
    freedofs_forces = forces[freedofs.cpu().numpy()]

    # Calculate u_nonzero
    keep_k_entries = k_entries[keep]
    u_nonzero = utils.solve_coo(
        keep_k_entries, indices, freedofs_forces, sym_pos=False
    )
    u_nonzero = u_nonzero.to(device=device, dtype=dtype)
    u_values = torch.cat((u_nonzero, torch.zeros(len(fixdofs)).to(device=device, dtype=dtype)))

    return u_values[index_map.cpu().numpy()]


def get_KU(x_phys, ke, freedofs, fixdofs, *, penal=3, e_min=1e-9, e_0=1,device=torch.device('cpu'), dtype=torch.double):
    """
    Function that displaces the load x using finite element techniques.
    """
    stiffness = young_modulus(x_phys, e_0, e_min, p=penal)

    # Get the K values
    k_entries, k_ylist, k_xlist = get_k(stiffness, ke)
    k_ylist = k_ylist.to(device=device, dtype=dtype)
    k_xlist = k_xlist.to(device=device, dtype=dtype)

    index_map, keep, indices = utils._get_dof_indices(
        freedofs, fixdofs, k_ylist, k_xlist
    )

    # Reduced forces
    # freedofs_forces = forces[freedofs.cpu().numpy()]

    # Calculate u_nonzero
    keep_k_entries = k_entries[keep]

    K = torch.sparse_coo_tensor(indices, keep_k_entries, (torch.numel(freedofs),) * 2)
    K = (K + K.transpose(1, 0)) / 2.0

    return K, index_map.cpu().numpy()    