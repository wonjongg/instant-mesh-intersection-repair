#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def total_volume(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute the total signed volume of a triangulated mesh.

    Args:
        vertices: Tensor of shape (V, 3) containing vertex coordinates
        faces: Long tensor of shape (F, 3) containing face indices

    Returns:
        Scalar tensor containing the total signed volume
    """
    # Get vertices of each face
    face_vertices = vertices[faces]  # Shape: (F, 3, 3)

    # Extract individual vertices
    v0 = face_vertices[:, 0]  # Shape: (F, 3)
    v1 = face_vertices[:, 1]  # Shape: (F, 3)
    v2 = face_vertices[:, 2]  # Shape: (F, 3)

    # Compute cross product of edges
    cross_product = torch.cross(v1 - v0, v2 - v0)  # Shape: (F, 3)

    # Dot product with first vertex
    volumes = torch.sum(v0 * cross_product, dim=1)  # Shape: (F,)

    # Sum and divide by 6 for total volume
    total_volume = torch.sum(volumes) / 6.0  # Scalar

    return total_volume

def volume_jacobian(vertices: torch.Tensor, faces: torch.Tensor, normalize=True) -> torch.Tensor:
    """
    Compute the Jacobian of the volume constraint with respect to vertex positions.
    
    Args:
        vertices: Tensor of shape (V, 3) containing vertex coordinates
        faces: Long tensor of shape (F, 3) containing face indices
        
    Returns:
        Tensor of shape (V, 3) containing the gradient for each vertex
    """
    # Initialize Jacobian
    jacobian = torch.zeros_like(vertices)  # Shape: (V, 3)
    
    # For each vertex in face, compute contribution
    for i in range(3):
        # Cyclically rotate indices for cross product
        j = (i + 1) % 3
        k = (i + 2) % 3
        
        # Get corresponding vertices
        vj = vertices[faces[:, j]]  # Next vertex
        vk = vertices[faces[:, k]]  # Previous vertex
        
        # Compute cross product contribution
        cross = torch.cross(vj, vk) / 6.0
        
        # Add contribution to corresponding vertex gradient
        jacobian.index_add_(0, faces[:, i], cross)
    
    if normalize:
        jacobian = jacobian / torch.maximum(torch.linalg.vector_norm(jacobian, dim=-1, keepdim=True), torch.tensor(1e-6).type_as(jacobian))

    return jacobian
