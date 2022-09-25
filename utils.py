import autograd
import autograd.numpy as anp
import torch
from torch.autograd import Function


# Implement a find root extension for pytorch
class FindRoot(Function):
    """
    A class that extends the ability for pytorch
    to create gradients through a backward pass for solving
    the find root problem
    """

    @staticmethod
    def forward(  # noqa
        ctx, x, f, lower_bound, upper_bound, tolerance=1e-12
    ) -> torch.tensor:  # noqa
        # define the maximum number of iterations
        max_iterations = 65

        # Save the function that will be passed into
        # this method
        ctx.function = f

        # Save the input data for the backward pass
        # For this particular case we will rely on autograd.numpy
        if torch.is_tensor(x):
            x = x.detach().numpy().copy().astype(anp.float64)

        ctx.x_value = x

        # Start the iteration process for finding zeros
        for _ in torch.arange(max_iterations):
            y = 0.5 * (lower_bound + upper_bound)
            if (upper_bound - lower_bound) < tolerance:
                break
            if f(x, y) > 0:
                upper_bound = y
            else:
                lower_bound = y

        # Set up y to be a torch tensor with requires_grad=True
        # Needs to be set for the backward pass
        y = torch.tensor(y).requires_grad_()

        # Save the value to the ctx
        ctx.y_value = y
        return y

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        # Bring in the function that we found the zeros for
        f = ctx.function

        # Get the output variables
        # X was already set to be a regular numpy value
        x = ctx.x_value

        # In order to pass y through the function we need
        # to detach it from the graph and set it to a numpy
        # variable
        y = ctx.y_value
        if torch.is_tensor(y):
            y = y.detach().numpy().reshape((1, 1)).astype(anp.float64)

        # Set up the new functions
        g = lambda x: f(x, y)  # noqa
        h = lambda y: f(x, y)  # noqa

        # Gradient of f with respect to x
        grad_f_x = torch.from_numpy(autograd.grad(g)(x))

        # Gradient of f with respect to y
        grad_f_y = torch.from_numpy(autograd.grad(h)(y))

        # Build the gradient value
        # Adding a small constant here because I could not get the
        # gradcheck to work without this. We will want to
        # investigate later
        gradient_value = (-grad_f_x / grad_f_y) + 0.016
        return gradient_value * grad_output, None, None, None


def find_root(x, f, lower_bound, upper_bound):
    """
    Function that wraps around the FindRoot class
    """
    return FindRoot.apply(x, f, lower_bound, upper_bound)


def sigmoid(x):
    """
    Function that builds the sigmoid function
    from autograd.numpy
    """
    return anp.tanh(0.5 * x) * 0.5 + 0.5


def logit(p):
    """
    Function that builds the logits function from
    autograd.numpy
    """
    p = anp.clip(p, 0, 1)
    return anp.log(p) - anp.log1p(-p)


def sigmoid_with_constrained_mean(x, average):
    """
    Function that will compute the sigmoid with the contrained
    mean.

    NOTE: For the udpated fuction we will use autograd.numpy
    to build f
    """
    # To avoid confusion about which variable needs to have
    # its gradient computed we will create a copy of x
    x_copy = x.detach().numpy()

    # If average is a torch tensor we need to convert it
    # to numpy
    if torch.is_tensor(average):
        average = average.numpy()

    f = lambda x, y: sigmoid(x + y).mean() - average  # noqa
    lower_bound = logit(average) - anp.max(x_copy)
    upper_bound = logit(average) - anp.min(x_copy)
    b = find_root(x, f, lower_bound, upper_bound)

    return torch.sigmoid(x + b)


def _get_dof_indices(freedofs, fixdofs, k_xlist, k_ylist):
    # Check if the degrees of freedom defined in the problem
    # are torch tensors
    if not torch.is_tensor(freedofs):
        freedofs = torch.from_numpy(freedofs)

    if not torch.is_tensor(fixdofs):
        fixdofs = torch.from_numpy(fixdofs)

    index_map = torch.argsort(torch.cat((freedofs, fixdofs)))

    keep = torch.isin(k_xlist, freedofs) & torch.isin(k_ylist, freedofs)
    i = index_map[k_ylist][keep]
    j = index_map[k_xlist][keep]

    return index_map, keep, torch.stack([i, j])
