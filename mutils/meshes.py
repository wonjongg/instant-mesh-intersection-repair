#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np 
import potpourri3d as pp3d
from scipy import sparse

def quad_to_tri(faces):
    f0, f1, f2, f3 = faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 3]

    tri1 = np.stack([f0, f1, f2], axis=1)
    tri2 = np.stack([f0, f2, f3], axis=1)

    tris = np.concatenate([tri1, tri2], axis=0)
    
    return tris

def cotan_laplacian(verts, faces):
    lap = pp3d.cotan_laplacian(verts, faces)
    coo = lap.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    shape = coo.shape
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    lap = torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce()

    return lap

def curv_laplacian(verts, faces, alpha=1.0, beta=2.0, min_weights=1e-6, epsilon=1e-1):
    # 기본 cotangent Laplacian 계산
    lap = pp3d.cotan_laplacian(verts, faces)
    curv = lap.dot(verts)
    curv = np.sqrt(np.sum(curv**2, axis=1))
    
    # sparse matrix 분해
    rows, cols = lap.nonzero()
    values = lap.data
    N = verts.shape[0]
    
    # edge curvature 계산
    edge_curv = (curv[rows] + curv[cols]) / 2
    
    # weight 조정 - exponential decay 사용
    weights = np.exp(-alpha * edge_curv)  # 항상 0과 1 사이의 값
    weights = np.clip(weights, min_weights, 1.0)  # numerical stability를 위한 최소값 설정
    
    # original Laplacian의 구조를 유지하면서 weight만 조정
    weighted_values = values * weights
    
    # symmetric matrix 보장
    weighted_lap = sparse.csr_matrix((weighted_values, (rows, cols)), shape=(N, N))
    weighted_lap = 0.5 * (weighted_lap + weighted_lap.T)
    
    # diagonal 재계산
    diag = -weighted_lap.sum(axis=1).A.flatten()
    
    # numerical stability를 위한 작은 shift 추가
    shift = epsilon * np.abs(diag).max()
    diag += shift
    
    # final matrix 구성
    weighted_lap = weighted_lap + sparse.diags([diag], [0])
    
    # torch sparse tensor로 변환
    coo = weighted_lap.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    shape = coo.shape
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    lap = torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce()
    
    return lap
    '''
    weights = np.exp(-alpha * edge_curv) ** beta
    weights = np.maximum(weights, min_weights)
    print(weights.max())
    print(weights.min())

    signs = np.sign(values)
    weighted_values = signs * np.abs(values) * weights
    print(f'max: {weighted_values.max()}')
    print(f'min: {weighted_values.min()}')

    N = verts.shape[0]
    curv_lap = sparse.csr_matrix((weighted_values, (rows, cols)), shape=(N, N))
    curv_lap = 0.5 * (curv_lap + curv_lap.T)

    print(curv_lap.todense()[10:, 10:])

    curv_lap = lap * curv_lap

    diag = -curv_lap.sum(axis=1).A.flatten()
    shift = epsilon * np.abs(diag).max()
    diag += shift
    #  curv_lap = curv_lap + sparse.diags([diag], [0])

    #  coo = curv_lap.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    shape = coo.shape
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    lap = torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce()

    indices = np.vstack((rows[mask], cols[mask]))
    values = values[mask]
    shape = lap.shape
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    lap = torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce()
    return lap
    '''

def triangle_vertex_incidence_matrix(verts, faces):
    '''
    Args:
        verts (np.ndarray, shape of (#V, 3))
        faces (np.ndarray, shape of (#F, 3))

    Returns:
        B (np.ndarray, shape of (#F, #V))
    '''
    num_verts = len(verts)
    num_faces = len(faces)

    B = np.zeros((num_faces, num_verts))

    values = np.full(faces.shape, 1/3)

    np.add.at(B, (np.repeat(np.arange(num_faces), 3), faces.flatten()), values.flatten())

    return B

def vertex_areas(face_areas, angles, vertices, faces, method='barycentric'):
    """
    Compute vertex areas from face areas.
    
    Args:
        vertices: Tensor of shape (V, 3) containing vertex positions
        method: Area distribution method ['uniform', 'barycentric']
    
    Returns:
        vertex_areas: Tensor of shape (V,) containing vertex areas
    """
    device = vertices.device
    V = vertices.shape[0]
    
    # Initialize vertex areas
    vertex_areas = torch.zeros(V, dtype=torch.float32, device=device)
    
    if method == 'uniform':
        # Distribute face area equally to each vertex (1/3)
        face_areas_per_vertex = face_areas.unsqueeze(1).expand(-1, 3) / 3.0
        vertex_areas.index_add_(0, faces.view(-1), face_areas_per_vertex.view(-1))
        
    elif method == 'barycentric':
        # Normalize angles to get weights
        weights = angles / angles.sum(dim=1, keepdim=True)
        
        # Distribute face areas according to weights
        face_areas_per_vertex = face_areas.unsqueeze(1) * weights
        vertex_areas.index_add_(0, faces.view(-1), face_areas_per_vertex.view(-1))
        
    else:
        raise ValueError(f"Unknown distribution method: {method}")
    
    return vertex_areas

def vertex_normals(face_normals, face_attributes, vertices, faces, method='angle_weighted'):
    """
    Compute vertex normals from face normals using vectorized operations.
    
    Args:
        vertices: Tensor of shape (V, 3) containing vertex positions
        method: Weighting method ['uniform', 'area_weighted', 'angle_weighted']
    
    Returns:
        vertex_normals: Tensor of shape (V, 3) containing vertex normals
    """
    V = vertices.shape[0]
    
    # Initialize vertex normals
    vertex_normals = torch.zeros((V, 3)).type_as(vertices)
    
    # Expand face normals for each vertex in the face (F, 3, 3)
    face_normals_expanded = face_normals.unsqueeze(1).expand(-1, 3, -1)
    
    if method == 'uniform':
        # Simple vectorized averaging
        vertex_normals.index_add_(0, faces.view(-1), face_normals_expanded.reshape(-1, 3))
        
    elif method == 'area_weighted':
        # Compute areas and weight the normals
        weighted_normals = face_normals_expanded * face_attributes.unsqueeze(-1).unsqueeze(-1)
        vertex_normals.index_add_(0, faces.view(-1), weighted_normals.reshape(-1, 3))
        
    elif method == 'angle_weighted':
        # Compute angles and weight the normals
        weighted_normals = face_normals_expanded * face_attributes.unsqueeze(-1)
        vertex_normals.index_add_(0, faces.view(-1), weighted_normals.reshape(-1, 3))
        
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize the vertex normals
    vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=1)
    
    return vertex_normals

def vertex_angles(vertices, faces):
    """
    Vectorized computation of angles at each vertex.
    """
    # Get vertices for each face (F, 3, 3)
    face_vertices = vertices[faces]
    
    # Compute edge vectors
    v0, v1, v2 = face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2]
    e0 = v1 - v0  # (F, 3)
    e1 = v2 - v1  # (F, 3)
    e2 = v0 - v2  # (F, 3)
    
    # Normalize edge vectors
    e0_norm = torch.nn.functional.normalize(e0, p=2, dim=1)
    e1_norm = torch.nn.functional.normalize(e1, p=2, dim=1)
    e2_norm = torch.nn.functional.normalize(e2, p=2, dim=1)
    
    # Compute all angles at once
    angles = torch.empty((faces.shape[0], 3)).type_as(vertices)
    angles[:, 0] = torch.acos(torch.clamp(-torch.sum(e0_norm * e2_norm, dim=1), -1.0, 1.0))
    angles[:, 1] = torch.acos(torch.clamp(-torch.sum(e1_norm * e0_norm, dim=1), -1.0, 1.0))
    angles[:, 2] = torch.acos(torch.clamp(-torch.sum(e2_norm * e1_norm, dim=1), -1.0, 1.0))
    
    return angles
