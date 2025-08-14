#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import random
import argparse
import numpy as np
import torch
import potpourri3d as pp3d

from glob import glob 

from mesh_intersection.bvh_search_tree import BVH

from largesteps.geometry import compute_matrix_from_lap
from largesteps.parameterize import from_differential, to_differential
from largesteps.solvers import CholeskySolver, solve
from largesteps.optimize import AdamUniform

import constraints
import mutils
import energies
import configs
import optimizers

def main(config):
    np_v, np_f = pp3d.read_mesh(config['objpath'])
    v_range = np_v.max() - np_v.min()
    np_v = 16.5 * np_v / v_range
    q = np_f
    if np_f.shape[1] == 4:
        np_f = mutils.quad_to_tri(np_f)
    pp3d.write_mesh(np_v, np_f, f'{config["savepath"]}/{config["expname"]}_init.obj')
    print('Number of triangles = ', np_f.shape[0])

    vertices = torch.tensor(np_v,
                            dtype=torch.float32, device=device).cuda()
    faces = torch.tensor(np_f,
                         dtype=torch.long,
                         device=device).cuda()

    search_tree = BVH(max_collisions=config['max_collisions'])

    if config['lap'] == 'cotan':
        lap = mutils.cotan_laplacian(np_v, np_f).float().cuda()
    elif config['lap'] == 'curv':
        lap = mutils.curv_laplacian(np_v, np_f).float().cuda()
    else:
        raise NotImplementedError(f'Not implemented Laplacian type: {config["lap"]}')

    M = compute_matrix_from_lap(lap, vertices, lambda_=0, alpha=0.99)

    u = to_differential(M, vertices)
    u.requires_grad = True

    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam([u], lr=config['lr'])
    elif config['optimizer'] == 'GD':
        optimizer = optimizers.GradientDescent([u], lr=config['lr'])
    elif config['optimizer'] == 'MomentumBrake':
        optimizer = optimizers.MomentumBrake([u], lr=config['lr'])
    elif config['optimizer'] == 'AdamUniform':
        optimizer = AdamUniform([u], lr=config['lr'])
    else:
        raise NotImplementedError(f'Not implemented optimizer type: {config["optimizer"]}')

    with torch.no_grad():
        A0 = constraints.total_area(vertices, faces)
        V0 = constraints.total_volume(vertices, faces)
        L0 = lap @ vertices

        triangles = vertices[faces]
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]


    if config['timing']:
        torch.cuda.synchronize()
        start = time.time()
    for i in range(60):
        optimizer.zero_grad()
        vertices = from_differential(M, u, 'Cholesky')

        if config['energy'] == 'signed_TPE':
            pen_loss = energies.signed_TPE(vertices, faces, search_tree, p=3)
        elif config['energy'] == 'signed_TPE_verts':
            pen_loss = energies.signed_TPE_verts(vertices, faces, search_tree)
        elif config['energy'] == 'TPE':
            pen_loss = energies.TPE(vertices, faces, search_tree)
        elif config['energy'] == 'p2plane':
            pen_loss = energies.p2plane(vertices, faces, search_tree)
        elif config['energy'] == 'conical':
            pen_loss = energies.conical(vertices, faces, search_tree)
        else:
            raise NotImplementedError('Not implemented energy')

        reg_loss = torch.tensor(0.0).cuda()
        if 'volume' in config['constraints']:
            reg_loss += torch.nn.functional.l1_loss(constraints.total_volume(vertices, faces), V0)
        if 'area' in config['constraints']:
            reg_loss += torch.nn.functional.l1_loss(constraints.total_area(vertices, faces), A0)
        if 'curvature' in config['constraints']:
            reg_loss += 1e6 * torch.nn.functional.mse_loss(lap @ vertices, L0)

        loss = pen_loss + reg_loss
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            if num_col < best_col:
                best_col = num_col
                best_vertices = vertices.detach().clone()
                best_iter = i

        if (i + 1) % 10 == 0:
            print(pen_loss.item(), reg_loss.item())
            with torch.no_grad():
                pp3d.write_mesh(vertices.detach().cpu().numpy(), np_f, f'{config["savepath"]}/{config["expname"]}_{(i+1):03d}.obj')

    if q is not None:
        pp3d.write_mesh(best_vertices.cpu().numpy(), q, f'{config["savepath"]}/{fname}_best.obj')
        pp3d.write_mesh(vertices.detach().cpu().numpy(), q, f'{config["savepath"]}/{fname}_final.obj')
    else:
        pp3d.write_mesh(best_vertices.cpu().numpy(), np_f, f'{config["savepath"]}/{fname}_best.obj')
        pp3d.write_mesh(vertices.detach().cpu().numpy(), np_f, f'{config["savepath"]}/{fname}_final.obj')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = configs.load_config(args.config)
    configs.save_experiment_config(config)
    device = torch.device('cuda')

    main(config)
