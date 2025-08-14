#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def total_area(verts, faces):
    # 삼각형의 세 정점을 가져옵니다
    v0 = verts[faces[:, 0]]  # (F, 3)
    v1 = verts[faces[:, 1]]  # (F, 3) 
    v2 = verts[faces[:, 2]]  # (F, 3)
    
    # 삼각형의 두 엣지 벡터를 계산합니다
    e1 = v1 - v0  # (F, 3)
    e2 = v2 - v0  # (F, 3)
    
    # 외적을 이용해 면적을 계산합니다
    normals = torch.cross(e1, e2, dim=1)  # (F, 3)
    areas = 0.5 * torch.norm(normals, dim=1)  # (F,)
    
    # 전체 면적을 구합니다
    total_area = torch.sum(areas)  # scalar
    
    return total_area

def area_jacobian(lap, verts, normalize=True):
    j = lap @ verts
    if normalize:
        j = j / torch.maximum(torch.linalg.vector_norm(j, dim=-1, keepdim=True), torch.tensor(1e-6).type_as(j))
    return j
