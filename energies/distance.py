#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def point_to_plane_distance(points, plane_points, plane_normals):
    """
    Calculate signed distance from points to planes.

    Args:
        points: (B, N, 3) or (N, 3) tensor of points
        plane_points: (B, M, 3) or (M, 3) tensor of points on planes
        plane_normals: (B, M, 3) or (M, 3) tensor of normalized plane normal vectors

    Returns:
        distances: (B, N) or (N, ) tensor of signed distances
    """
    # Ensure input tensors are at least 3D
    if points.dim() == 2:
        points = points.unsqueeze(0)
    if plane_points.dim() == 2:
        plane_points = plane_points.unsqueeze(0)
    if plane_normals.dim() == 2:
        plane_normals = plane_normals.unsqueeze(0)

    # Calculate vectors from plane points to query points
    # vectors: (B, N, 3)
    vectors = points - plane_points
    
    # Calculate distances using dot product
    # vectors: (B, N, 3)
    # plane_normals: (B, N, 3)
    # distances: (B, N)
    distances = torch.sum(vectors * plane_normals, dim=-1)

    # Remove batch dimension if input was 2D
    if points.size(0) == 1 and plane_points.size(0) == 1 and plane_normals.size(0) == 1:
        distances = distances.squeeze(0)

    return distances

