#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Conical penetration distance energy for mesh intersection repair.
"""

import torch
import mesh_intersection.loss as collisions_loss


def conical(verts, faces, search_tree, return_col=False):
    """
    Compute conical penetration distance energy.

    Uses the distance field penetration loss from the mesh_intersection library
    to penalize self-intersections with a conical distance formulation.

    Args:
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        faces (torch.Tensor): Face indices of shape (F, 3)
        search_tree (BVH): BVH search tree for collision detection
        return_col (bool, optional): Whether to return number of collisions. Defaults to False.

    Returns:
        torch.Tensor: Conical penetration energy scalar
        int (optional): Number of collisions if return_col=True
    """
    triangles = verts[faces]

    # Detect collisions using BVH search tree
    with torch.no_grad():
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]
        num_col = collision_idxs.shape[0]
        print(collision_idxs.shape)

    # Configure penetration loss parameters
    sigma = 0.5
    point2plane = True

    # Create distance field penetration loss
    pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
        sigma=sigma,
        point2plane=point2plane,
        vectorized=True
    )
    if return_col:
        return pen_distance(triangles.unsqueeze(0), collision_idxs.unsqueeze(0)), num_col
    else:
        return pen_distance(triangles.unsqueeze(0), collision_idxs.unsqueeze(0))
