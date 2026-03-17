"""
Area constraints and related functions for mesh optimization.
"""

import torch


def total_area(verts, faces):
    """
    Compute the total surface area of a triangulated mesh.

    Args:
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        faces (torch.Tensor): Face indices of shape (F, 3)

    Returns:
        torch.Tensor: Total surface area (scalar)
    """
    # Get vertices for each face
    v0 = verts[faces[:, 0]]  # (F, 3)
    v1 = verts[faces[:, 1]]  # (F, 3)
    v2 = verts[faces[:, 2]]  # (F, 3)

    # Compute edge vectors
    e1 = v1 - v0  # (F, 3)
    e2 = v2 - v0  # (F, 3)

    # Compute face areas using cross product
    normals = torch.cross(e1, e2, dim=1)  # (F, 3)
    areas = 0.5 * torch.norm(normals, dim=1)  # (F,)

    # Sum all face areas
    total_area = torch.sum(areas)  # scalar

    return total_area


def area_jacobian(lap, verts, normalize=True):
    """
    Compute the Jacobian of the area constraint using Laplacian regularization.

    Args:
        lap (torch.Tensor): Laplacian matrix
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        normalize (bool, optional): Whether to normalize the Jacobian. Defaults to True.

    Returns:
        torch.Tensor: Jacobian of shape (V, 3)
    """
    # Compute Laplacian-based Jacobian
    j = lap @ verts

    # Normalize if requested
    if normalize:
        j = j / torch.maximum(
            torch.linalg.vector_norm(j, dim=-1, keepdim=True),
            torch.tensor(1e-6).type_as(j)
        )

    return j
