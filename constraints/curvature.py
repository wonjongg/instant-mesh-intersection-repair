"""
Curvature constraints for mesh optimization.
"""

import torch


def meancurv_jacobian(lap, verts, u):
    """
    Compute the Jacobian of the mean curvature constraint.

    The mean curvature constraint penalizes deviation from an initial curvature
    distribution represented by the Laplacian coordinates.

    Args:
        lap (torch.Tensor): Laplacian matrix
        verts (torch.Tensor): Current vertex positions of shape (V, 3)
        u (torch.Tensor): Target Laplacian coordinates of shape (V, 3)

    Returns:
        torch.Tensor: Jacobian of the curvature constraint of shape (V, 3)
    """
    # Compute difference between current and target Laplacian coordinates
    Lx_diff = lap @ verts - u

    # Compute Jacobian as derivative of squared L2 norm
    jacobian = 2 * lap.T @ Lx_diff

    return jacobian
