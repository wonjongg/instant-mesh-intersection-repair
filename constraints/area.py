import torch

def total_area(verts, faces):
    v0 = verts[faces[:, 0]]  # (F, 3)
    v1 = verts[faces[:, 1]]  # (F, 3) 
    v2 = verts[faces[:, 2]]  # (F, 3)
    
    e1 = v1 - v0  # (F, 3)
    e2 = v2 - v0  # (F, 3)
    
    normals = torch.cross(e1, e2, dim=1)  # (F, 3)
    areas = 0.5 * torch.norm(normals, dim=1)  # (F,)
    
    total_area = torch.sum(areas)  # scalar
    
    return total_area

def area_jacobian(lap, verts, normalize=True):
    j = lap @ verts
    if normalize:
        j = j / torch.maximum(torch.linalg.vector_norm(j, dim=-1, keepdim=True), torch.tensor(1e-6).type_as(j))
    return j
