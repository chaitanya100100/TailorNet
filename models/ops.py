import torch
from global_var import VALID_THETA


def verts_dist(v1, v2, dim=None):
    """
    distance between two point sets
    v1 and v2 shape: NxVx3
    """
    x = torch.pow(v2 - v1, 2)
    x = torch.sum(x, -1)
    x = torch.sqrt(x)
    if dim == -1:
        return x
    elif dim is None:
        return torch.mean(x)
    else:
        return torch.mean(x, dim=dim)


def mask_thetas(thetas, garment_class):
    """
    thetas: shape [N, 72]
    garment_class: e.g. t-shirt
    """
    valid_theta = VALID_THETA[garment_class]
    mask = torch.zeros_like(thetas).view(-1, 24, 3)
    mask[:, valid_theta, :] = 1.
    mask = mask.view(-1, 72)
    return thetas * mask


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist
