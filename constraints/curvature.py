import torch

def meancurv_jacobian(lap, verts, u):
    Lx_diff = lap @ verts - u
    
    jacobian = 2 * lap.T @ Lx_diff

    return jacobian
