#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triangle Proximity Energy (TPE) implementations.

This module provides various formulations of Triangle Proximity Energy for
penalizing mesh self-intersections.
"""

import torch
import time


def TPE(verts, faces, search_tree, return_col=False):
    """
    Compute Triangle Proximity Energy (TPE) for mesh self-intersections.

    TPE measures proximity between colliding triangles based on their face centers
    and normal-projected distances, weighted by face areas.

    Args:
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        faces (torch.Tensor): Face indices of shape (F, 3)
        search_tree (BVH): BVH search tree for collision detection
        return_col (bool, optional): Whether to return number of collisions. Defaults to False.

    Returns:
        torch.Tensor: TPE energy scalar
        int (optional): Number of collisions if return_col=True
    """
    triangles = verts[faces]

    # Detect collisions and precompute geometric properties
    with torch.no_grad():
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]
        print(collision_idxs.shape)
        num_col = collision_idxs.shape[0]

        # Compute face normals and areas
        face_normals = torch.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0]
        )
        face_areas = 0.5 * torch.norm(face_normals, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)

    # Compute face centers (centroids)
    face_centers = torch.sum(triangles, dim=1) / 3  # [NF, 3]

    # Get centers of colliding face pairs
    X = face_centers[collision_idxs[:, 0]]
    Y = face_centers[collision_idxs[:, 1]]

    # Compute distances and projected distances
    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.abs(torch.sum((X - Y) * face_normals[collision_idxs[:, 0]], dim=1))
    Pdist_y = torch.abs(torch.sum((Y - X) * face_normals[collision_idxs[:, 1]], dim=1))

    # Compute ratios (cubed for stronger penalty)
    r_x = (Pdist_x / dist).pow(3)
    r_y = (Pdist_y / dist).pow(3)

    # Compute TPE weighted by geometric mean of face areas
    TPE = torch.sqrt(face_areas[collision_idxs[:, 0]] * face_areas[collision_idxs[:, 1]]) * (r_x + r_y)
    TPE = torch.sum(TPE)

    if return_col:
        return TPE, num_col
    else:
        return TPE

def signed_TPE(verts, faces, search_tree, p=3):
    """
    Compute signed Triangle Proximity Energy with configurable power.

    Unlike standard TPE, this variant uses signed projected distances (without absolute value)
    to allow for directional penalties.

    Args:
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        faces (torch.Tensor): Face indices of shape (F, 3)
        search_tree (BVH): BVH search tree for collision detection
        p (int, optional): Power for distance ratio. Defaults to 3.

    Returns:
        torch.Tensor: Signed TPE energy scalar
    """
    triangles = verts[faces]
    with torch.no_grad():
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]
        print(collision_idxs.shape)
        face_normals = torch.cross(triangles[:, 1] - triangles[:, 0], \
                                        triangles[:, 2] - triangles[:, 0])
        face_areas = 0.5 * torch.norm(face_normals, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)

    face_centers = torch.sum(triangles, dim=1) / 3 # [NF, 3]

    X = face_centers[collision_idxs[:, 0]]
    Y = face_centers[collision_idxs[:, 1]]

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((X - Y) * face_normals[collision_idxs[:, 0]], dim=1)
    Pdist_y = torch.sum((Y - X) * face_normals[collision_idxs[:, 1]], dim=1)

    r_x = (Pdist_x / dist).pow(p)
    r_y = (Pdist_y / dist).pow(p)

    TPE = face_areas[collision_idxs[:, 0]] * face_areas[collision_idxs[:, 1]] * (r_x + r_y)
    TPE = torch.sum(TPE)

    return TPE

def signed_TPE_verts(verts, faces, search_tree, p=1, return_col=False):
    """
    Compute signed TPE using vertex-to-face-center distances.

    Instead of using face center pairs, this formulation measures distances from
    individual vertices of one face to the center of the colliding face.

    Args:
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        faces (torch.Tensor): Face indices of shape (F, 3)
        search_tree (BVH): BVH search tree for collision detection
        p (int, optional): Power for distance ratio. Defaults to 1.
        return_col (bool, optional): Whether to return number of collisions. Defaults to False.

    Returns:
        torch.Tensor: Signed TPE energy scalar
        int (optional): Number of collisions if return_col=True
    """
    triangles = verts[faces]
    with torch.no_grad():
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]
        num_col = collision_idxs.shape[0]
        face_normals = torch.cross(triangles[:, 1] - triangles[:, 0], \
                                        triangles[:, 2] - triangles[:, 0])
        face_areas = 0.5 * torch.norm(face_normals, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)

    face_centers = torch.sum(triangles, dim=1) / 3 # [NF, 3]

    verts_idx = faces[collision_idxs[:, 0]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 1]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 1]].repeat(3, 1), dim=1)
    r_x = (Pdist_x / dist).pow(p)
    #  r_x = (Pdist_x ).pow(p)
    TPE1 = face_areas[collision_idxs[:, 1]].repeat(3) * r_x

    verts_idx = faces[collision_idxs[:, 1]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 0]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 0]].repeat(3, 1), dim=1)
    r_x = (Pdist_x / dist).pow(p)
    #  r_x = (Pdist_x ).pow(p)
    TPE2 = face_areas[collision_idxs[:, 0]].repeat(3) * r_x

    TPE = torch.cat([TPE1, TPE2], dim=0)
    TPE = torch.sum(TPE)

    if return_col:
        return TPE, num_col
    else:
        return TPE

def signed_TPE_verts_mask(verts, faces, search_tree, mask1, mask2, p=3, return_col=False):
    """
    Compute signed TPE with masked face normals.

    This variant allows masking certain faces to use custom normals instead of
    computed face normals, useful for handling special geometric configurations.

    Args:
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        faces (torch.Tensor): Face indices of shape (F, 3)
        search_tree (BVH): BVH search tree for collision detection
        mask1 (torch.Tensor): Boolean mask for first set of faces
        mask2 (torch.Tensor): Boolean mask for second set of faces
        p (int, optional): Power for distance ratio. Defaults to 3.
        return_col (bool, optional): Whether to return number of collisions. Defaults to False.

    Returns:
        torch.Tensor: Signed TPE energy scalar
        int (optional): Number of collisions if return_col=True
    """
    triangles = verts[faces]
    with torch.no_grad():
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]
        #  print(collision_idxs.shape)
        num_col = collision_idxs.shape[0]
        face_normals = torch.cross(triangles[:, 1] - triangles[:, 0], \
                                        triangles[:, 2] - triangles[:, 0])
        face_areas = 0.5 * torch.norm(face_normals, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        face_normals[mask1] = torch.zeros_like(face_normals[mask1])
        face_normals[mask1, 2] = 1.0
        face_normals[mask2] = torch.ones_like(face_normals[mask2])
        face_normals[mask2, 2] = 1.0

    face_centers = torch.sum(triangles, dim=1) / 3 # [NF, 3]

    verts_idx = faces[collision_idxs[:, 0]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 1]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 1]].repeat(3, 1), dim=1)
    r_x = (Pdist_x / dist).pow(p)
    #  r_x = (Pdist_x ).pow(p)
    TPE1 = face_areas[collision_idxs[:, 1]].repeat(3) * r_x

    verts_idx = faces[collision_idxs[:, 1]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 0]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 0]].repeat(3, 1), dim=1)
    r_x = (Pdist_x / dist).pow(p)
    #  r_x = (Pdist_x ).pow(p)
    TPE2 = face_areas[collision_idxs[:, 0]].repeat(3) * r_x

    TPE = torch.cat([TPE1, TPE2], dim=0)
    TPE = torch.sum(TPE)

    if return_col:
        return TPE, num_col
    else:
        return TPE
def signed_TPE_verts_test(verts, faces, collision_idxs, p=3):
    """
    Compute signed TPE for testing with precomputed collision indices.

    This is a testing variant that takes collision indices directly instead of
    computing them via BVH search.

    Args:
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        faces (torch.Tensor): Face indices of shape (F, 3)
        collision_idxs (torch.Tensor): Precomputed collision indices of shape (C, 2)
        p (int, optional): Power for distance ratio. Defaults to 3.

    Returns:
        torch.Tensor: Signed TPE energy scalar
    """
    triangles = verts[faces]
    with torch.no_grad():
        face_normals = torch.cross(triangles[:, 1] - triangles[:, 0], \
                                        triangles[:, 2] - triangles[:, 0])
        face_areas = 0.5 * torch.norm(face_normals, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)

    face_centers = torch.sum(triangles, dim=1) / 3 # [NF, 3]

    verts_idx = faces[collision_idxs[:, 0]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 1]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 1]].repeat(3, 1), dim=1)
    r_x = (Pdist_x / dist).pow(p)
    #  r_x = (Pdist_x ).pow(p)
    TPE1 = face_areas[collision_idxs[:, 1]].repeat(3) * r_x

    verts_idx = faces[collision_idxs[:, 1]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 0]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 0]].repeat(3, 1), dim=1)
    r_x = (Pdist_x / dist).pow(p)
    #  r_x = (Pdist_x ).pow(p)
    TPE2 = face_areas[collision_idxs[:, 0]].repeat(3) * r_x

    TPE = torch.cat([TPE1, TPE2], dim=0)
    TPE = torch.sum(TPE)

    return TPE

def signed_TPE_twoobj(verts, faces, search_tree, p=3):
    """
    Compute signed TPE for two-object collision scenarios.

    Similar to signed_TPE_verts but designed for handling collisions between
    two separate objects merged into a single mesh.

    Args:
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        faces (torch.Tensor): Face indices of shape (F, 3)
        search_tree (BVH): BVH search tree for collision detection
        p (int, optional): Power for distance ratio. Defaults to 3.

    Returns:
        torch.Tensor: Signed TPE energy scalar
    """
    triangles = verts[faces]
    with torch.no_grad():
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]
        print(collision_idxs.shape)
        face_normals = torch.cross(triangles[:, 1] - triangles[:, 0], \
                                        triangles[:, 2] - triangles[:, 0])
        face_areas = 0.5 * torch.norm(face_normals, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)

        face_centers = torch.sum(triangles, dim=1) / 3 # [NF, 3]

    verts_idx = faces[collision_idxs[:, 0]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 1]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 1]].repeat(3, 1), dim=1)
    r_x = (Pdist_x / dist).pow(p)
    #  r_x = (Pdist_x ).pow(p)
    TPE1 = face_areas[collision_idxs[:, 1]].repeat(3) * r_x

    verts_idx = faces[collision_idxs[:, 1]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 0]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 0]].repeat(3, 1), dim=1)
    r_x = (Pdist_x / dist).pow(p)
    #  r_x = (Pdist_x ).pow(p)
    TPE2 = face_areas[collision_idxs[:, 0]].repeat(3) * r_x

    TPE = torch.cat([TPE1, TPE2], dim=0)
    TPE = torch.sum(TPE)

    return TPE

def p2plane(verts, faces, search_tree, p=1, return_col=False):
    """
    Compute point-to-plane penetration energy.

    This formulation measures vertex-to-face-center distances projected onto a
    fixed plane normal (z-axis), useful for scenarios with known penetration direction.

    Args:
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        faces (torch.Tensor): Face indices of shape (F, 3)
        search_tree (BVH): BVH search tree for collision detection
        p (int, optional): Power for projected distance. Defaults to 1.
        return_col (bool, optional): Whether to return number of collisions. Defaults to False.

    Returns:
        torch.Tensor: Point-to-plane energy scalar
        int (optional): Number of collisions if return_col=True
    """
    triangles = verts[faces]
    with torch.no_grad():
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]
        num_col = collision_idxs.shape[0]
        print(collision_idxs.shape)
        face_normals = torch.cross(triangles[:, 1] - triangles[:, 0], \
                                        triangles[:, 2] - triangles[:, 0])
        face_areas = 0.5 * torch.norm(face_normals, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        face_normals = torch.tensor([0.0, 0.0, -1.0]).cuda().unsqueeze(0).repeat(face_normals.shape[0], 1)

    face_centers = torch.sum(triangles, dim=1) / 3 # [NF, 3]

    verts_idx = faces[collision_idxs[:, 0]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 1]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 1]].repeat(3, 1), dim=1)
    #  r_x = (Pdist_x / dist).pow(p)
    r_x = (Pdist_x ).pow(p)
    TPE1 = face_areas[collision_idxs[:, 1]].repeat(3) * r_x

    verts_idx = faces[collision_idxs[:, 1]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 0]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 0]].repeat(3, 1), dim=1)
    #  r_x = (Pdist_x / dist).pow(p)
    r_x = (Pdist_x ).pow(p)
    TPE2 = face_areas[collision_idxs[:, 0]].repeat(3) * r_x

    TPE = torch.cat([TPE1, TPE2], dim=0)
    TPE = torch.sum(TPE)

    if return_col:
        return TPE, num_col
    else:
        return TPE

