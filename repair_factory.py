#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Instant Self-Intersection Repair for 3D Meshes.

This module provides the main entry point for repairing self-intersecting
3D meshes using various energy-based optimization approaches with Laplacian
parameterization.
"""

import time
import argparse
from pathlib import Path
import numpy as np
import torch
import potpourri3d as pp3d

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
    """
    Main function for mesh intersection repair.

    This function loads a mesh, detects self-intersections using a BVH search tree,
    and optimizes the mesh to remove intersections while preserving geometric properties.

    Args:
        config (dict): Configuration dictionary containing:
            - objpath (str): Path to input mesh file
            - savepath (str): Directory to save output meshes
            - expname (str): Experiment name for output files
            - max_collisions (int): Maximum number of collisions to detect
            - optimizer (str): Optimizer type ('Adam', 'GD', 'MomentumBrake', 'AdamUniform')
            - lr (float): Learning rate
            - energy (str): Energy function type ('signed_TPE', 'signed_TPE_verts', 'TPE', 'p2plane', 'conical')
            - constraints (list): List of constraints to apply ('volume', 'area', 'curvature')

    Returns:
        None: Saves intermediate and final mesh files to disk.
    """
    # Ensure output directory exists
    Path(config['savepath']).mkdir(parents=True, exist_ok=True)

    # Load mesh and normalize it
    np_v, np_f = pp3d.read_mesh(config['objpath'])
    v_range = np_v.max() - np_v.min()
    np_v = 16.5 * np_v / v_range

    # Store original faces (quad faces if applicable)
    quad = np_f

    # Convert quad faces to triangles if necessary
    if np_f.shape[1] == 4:
        np_f = mutils.quad_to_tri(np_f)

    pp3d.write_mesh(np_v, np_f, f'{config["savepath"]}/{config["expname"]}_init.obj')
    print('Number of triangles = ', np_f.shape[0])

    # Convert to torch tensors
    vertices = torch.tensor(np_v, dtype=torch.float32, device=device).cuda()
    faces = torch.tensor(np_f, dtype=torch.long, device=device).cuda()

    # Initialize BVH search tree for collision detection
    search_tree = BVH(max_collisions=config['max_collisions'])

    # Compute Laplacian matrix based on specified type
    lap = mutils.cotan_laplacian(np_v, np_f).float().cuda()

    # Compute parameterization matrix M = (1-alpha)*I + alpha*L
    M = compute_matrix_from_lap(lap, vertices, lambda_=0, alpha=0.99)

    # Convert to differential coordinates
    u = to_differential(M, vertices)
    u.requires_grad = True

    # Initialize optimizer based on configuration
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

    # Compute initial geometric properties for constraints
    with torch.no_grad():
        if 'volume' in config['constraints']:
            V0 = constraints.total_volume(vertices, faces)
        if 'area' in config['constraints']:
            A0 = constraints.total_area(vertices, faces)
        if 'curvature' in config['constraints']:
            L0 = lap @ vertices

        triangles = vertices[faces]
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]

    # Initialize tracking variables for best solution
    best_col = float('inf')
    best_vertices = vertices.detach().clone()
    best_iter = 0
    fname = config['expname']

    # Optimization loop
    for i in range(60):
        optimizer.zero_grad()

        # Convert from differential to Cartesian coordinates
        vertices = from_differential(M, u, 'Cholesky')

        # Compute penetration energy based on selected method
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

        # Compute regularization losses for constraints
        reg_loss = torch.tensor(0.0).cuda()
        if 'volume' in config['constraints']:
            reg_loss += torch.nn.functional.l1_loss(constraints.total_volume(vertices, faces), V0)
        if 'area' in config['constraints']:
            reg_loss += torch.nn.functional.l1_loss(constraints.total_area(vertices, faces), A0)
        if 'curvature' in config['constraints']:
            reg_loss += 1e6 * torch.nn.functional.mse_loss(lap @ vertices, L0)

        # Backpropagate and update parameters
        loss = pen_loss + reg_loss
        loss.backward()
        optimizer.step()

        # Track best solution with minimum collisions
        with torch.no_grad():
            triangles = vertices[faces]
            collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
            collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]
            num_col = collision_idxs.shape[0]

            if num_col < best_col:
                best_col = num_col
                best_vertices = vertices.detach().clone()
                best_iter = i

        # Print progress and save intermediate results
        if (i + 1) % 10 == 0:
            print(f'Iteration {i+1}: Penetration Loss = {pen_loss.item():.6f}, Regularization Loss = {reg_loss.item():.6f}, Collisions = {num_col}')
            with torch.no_grad():
                pp3d.write_mesh(vertices.detach().cpu().numpy(), np_f, f'{config["savepath"]}/{config["expname"]}_{(i+1):03d}.obj')

    # Save final results
    print(f'\nOptimization complete. Best solution found at iteration {best_iter} with {best_col} collisions.')

    # Save best and final meshes (use original quad faces if available)
    if quad is not None and quad.shape[1] == 4:
        pp3d.write_mesh(best_vertices.cpu().numpy(), quad, f'{config["savepath"]}/{fname}_best.obj')
        pp3d.write_mesh(vertices.detach().cpu().numpy(), quad, f'{config["savepath"]}/{fname}_final.obj')
    else:
        pp3d.write_mesh(best_vertices.cpu().numpy(), np_f, f'{config["savepath"]}/{fname}_best.obj')
        pp3d.write_mesh(vertices.detach().cpu().numpy(), np_f, f'{config["savepath"]}/{fname}_final.obj')


def main_vis(config):
    """
    Visualization-enabled mesh repair using Polyscope.

    Opens an interactive Polyscope window that displays the mesh and
    collision highlights. Provides two UI buttons:
      - Step : execute exactly one optimization iteration
      - Run  : continuously execute iterations until complete (auto-play)
               clicking Run again while running pauses it

    Collision intensity is visualized as a per-face scalar (red = more collisions).
    The window can be closed at any time; best/final meshes are saved on completion.

    Args:
        config (dict): Same configuration dictionary as main().
    """
    import polyscope as ps
    import polyscope.imgui as psim

    MAX_STEPS = 60

    # Ensure output directory exists
    Path(config['savepath']).mkdir(parents=True, exist_ok=True)

    # ---- Load and normalize mesh ----
    np_v, np_f = pp3d.read_mesh(config['objpath'])
    v_range = np_v.max() - np_v.min()
    np_v = 16.5 * np_v / v_range
    quad = np_f.copy()
    if np_f.shape[1] == 4:
        np_f = mutils.quad_to_tri(np_f)
    print('Number of triangles = ', np_f.shape[0])

    # ---- Build torch tensors ----
    vertices = torch.tensor(np_v, dtype=torch.float32, device=device)
    faces    = torch.tensor(np_f, dtype=torch.long,    device=device)
    search_tree = BVH(max_collisions=config['max_collisions'])

    # ---- Laplacian ----
    lap = mutils.cotan_laplacian(np_v, np_f).float().to(device)

    M = compute_matrix_from_lap(lap, vertices, lambda_=0, alpha=0.99)
    u = to_differential(M, vertices)
    u.requires_grad = True

    # ---- Optimizer ----
    if config['optimizer'] == 'Adam':
        opt = torch.optim.Adam([u], lr=config['lr'])
    elif config['optimizer'] == 'GD':
        opt = optimizers.GradientDescent([u], lr=config['lr'])
    elif config['optimizer'] == 'MomentumBrake':
        opt = optimizers.MomentumBrake([u], lr=config['lr'])
    elif config['optimizer'] == 'AdamUniform':
        opt = AdamUniform([u], lr=config['lr'])
    else:
        raise NotImplementedError(f'Not implemented optimizer type: {config["optimizer"]}')

    # ---- Initial constraint reference values ----
    with torch.no_grad():
        V0 = constraints.total_volume(vertices, faces) if 'volume'    in config['constraints'] else None
        A0 = constraints.total_area(vertices, faces)   if 'area'      in config['constraints'] else None
        L0 = (lap @ vertices).clone()                  if 'curvature' in config['constraints'] else None

        init_tri     = vertices[faces]
        init_col_idx = search_tree(init_tri.unsqueeze(0)).squeeze(0)
        init_col_idx = init_col_idx[init_col_idx[:, 0] >= 0, :]

    # ---- Mutable optimization state (dict avoids 'nonlocal' everywhere) ----
    state = {
        'step'     : 0,
        'running'  : False,
        'done'     : False,
        'vertices' : vertices.detach().clone(),   # latest decoded vertices (for saving)
        'pen_loss' : 0.0,
        'reg_loss' : 0.0,
        'num_col'  : init_col_idx.shape[0],
        'best_col' : init_col_idx.shape[0],
        'best_verts': vertices.detach().clone(),
        'best_iter': 0,
    }

    # ---- Polyscope initialisation ----
    ps.init()
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh(
        "mesh", np_v, np_f, smooth_shade=True, enabled=True
    )

    def _collision_mask(col_idxs):
        """Return a per-face array counting how many collision pairs involve each face."""
        mask = np.zeros(np_f.shape[0], dtype=np.float32)
        if col_idxs.shape[0] > 0:
            cf = col_idxs.cpu().numpy().flatten()
            cf = cf[cf >= 0]
            np.add.at(mask, cf, 1.0)
        return mask

    # Show initial collision state
    ps_mesh.add_scalar_quantity(
        "collisions", _collision_mask(init_col_idx),
        defined_on='faces', enabled=True, cmap='reds'
    )

    # ---- Single optimization step ----
    def do_step():
        if state['done']:
            return

        i = state['step']
        opt.zero_grad()

        # Decode from differential coordinates
        verts = from_differential(M, u, 'Cholesky')

        # Penetration energy
        if config['energy'] == 'signed_TPE':
            pen_loss = energies.signed_TPE(verts, faces, search_tree, p=3)
        elif config['energy'] == 'signed_TPE_verts':
            pen_loss = energies.signed_TPE_verts(verts, faces, search_tree)
        elif config['energy'] == 'TPE':
            pen_loss = energies.TPE(verts, faces, search_tree)
        elif config['energy'] == 'p2plane':
            pen_loss = energies.p2plane(verts, faces, search_tree)
        elif config['energy'] == 'conical':
            pen_loss = energies.conical(verts, faces, search_tree)
        else:
            raise NotImplementedError('Not implemented energy')

        # Regularization
        reg_loss = torch.tensor(0.0, device=device)
        if V0 is not None:
            reg_loss += torch.nn.functional.l1_loss(constraints.total_volume(verts, faces), V0)
        if A0 is not None:
            reg_loss += torch.nn.functional.l1_loss(constraints.total_area(verts, faces), A0)
        if L0 is not None:
            reg_loss += 1e6 * torch.nn.functional.mse_loss(lap @ verts, L0)

        loss = pen_loss + reg_loss
        loss.backward()
        opt.step()

        # Evaluate new state after parameter update
        with torch.no_grad():
            verts_new  = from_differential(M, u, 'Cholesky')
            triangles  = verts_new[faces]
            col_idxs   = search_tree(triangles.unsqueeze(0)).squeeze(0)
            col_idxs   = col_idxs[col_idxs[:, 0] >= 0, :]
            num_col    = col_idxs.shape[0]

            if num_col < state['best_col']:
                state['best_col']   = num_col
                state['best_verts'] = verts_new.detach().clone()
                state['best_iter']  = i

        # Update mutable state
        state['vertices']  = verts_new
        state['pen_loss']  = pen_loss.item()
        state['reg_loss']  = reg_loss.item()
        state['num_col']   = num_col
        state['step']     += 1

        # Refresh polyscope mesh and collision overlay
        ps_mesh.update_vertex_positions(verts_new.detach().cpu().numpy())
        ps_mesh.add_scalar_quantity(
            "collisions", _collision_mask(col_idxs),
            defined_on='faces', enabled=True, cmap='reds'
        )

        print(f'Step {state["step"]:3d}/{MAX_STEPS}  '
              f'pen={pen_loss.item():.6f}  reg={reg_loss.item():.6f}  col={num_col}')

        # Mark done and save when all steps are finished
        if state['step'] >= MAX_STEPS:
            state['done']    = True
            state['running'] = False
            print(f'\nOptimization complete. '
                  f'Best at step {state["best_iter"]} with {state["best_col"]} collisions.')
            _save_results()

    # ---- Save best and final meshes ----
    def _save_results():
        fname   = config['expname']
        best_v  = state['best_verts'].cpu().numpy()
        final_v = state['vertices'].detach().cpu().numpy()
        if quad.shape[1] == 4:
            pp3d.write_mesh(best_v,  quad, f'{config["savepath"]}/{fname}_best.obj')
            pp3d.write_mesh(final_v, quad, f'{config["savepath"]}/{fname}_final.obj')
        else:
            pp3d.write_mesh(best_v,  np_f, f'{config["savepath"]}/{fname}_best.obj')
            pp3d.write_mesh(final_v, np_f, f'{config["savepath"]}/{fname}_final.obj')

    # ---- Polyscope UI callback (called every frame) ----
    def callback():
        # --- Stats panel ---
        psim.TextUnformatted(f"Step       : {state['step']} / {MAX_STEPS}")
        psim.TextUnformatted(f"Collisions : {state['num_col']}")
        psim.TextUnformatted(f"Pen Loss   : {state['pen_loss']:.6f}")
        psim.TextUnformatted(f"Reg Loss   : {state['reg_loss']:.6f}")
        psim.TextUnformatted(f"Best       : step={state['best_iter']},  col={state['best_col']}")

        psim.Separator()

        if state['done']:
            psim.TextUnformatted("Optimization complete!")
            return

        # --- Step button: advance exactly one iteration ---
        if psim.Button("Step"):
            do_step()

        psim.SameLine()

        # --- Run / Pause toggle: auto-play or pause ---
        if state['running']:
            if psim.Button("Pause"):
                state['running'] = False
        else:
            if psim.Button("Run"):
                state['running'] = True

        # Auto-advance one step per frame while running
        if state['running']:
            do_step()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Instant Self-Intersection Repair for 3D Meshes'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--vis',
        action='store_true',
        default=False,
        help='Enable interactive Polyscope visualization with Step/Run buttons'
    )
    args = parser.parse_args()

    # Load configuration and setup device
    config = configs.load_config(args.config)
    configs.save_experiment_config(config)
    device = torch.device('cuda')

    # Run mesh repair (with or without visualization)
    if args.vis:
        main_vis(config)
    else:
        main(config)
