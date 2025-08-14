#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import mesh_intersection.loss as collisions_loss

def conical(verts, faces, search_tree, return_col=False):
    triangles = verts[faces]
    with torch.no_grad():
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]
        num_col = collision_idxs.shape[0]
        print(collision_idxs.shape)

    sigma = 0.5
    point2plane = True 

    pen_distance = \
        collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma,
                                                     point2plane=point2plane,
                                                     vectorized=True)
    if return_col:
        return pen_distance(triangles.unsqueeze(0), collision_idxs.unsqueeze(0)), num_col
    else:
        return pen_distance(triangles.unsqueeze(0), collision_idxs.unsqueeze(0))
