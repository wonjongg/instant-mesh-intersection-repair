# Instant Self-Intersection Repair for 3D Meshes

This repository contains the official implementation of the following paper:

> **[Instant Self-Intersection Repair for 3D Meshes (ACM TOG 2025)](https://dl.acm.org/doi/10.1145/3731427)**<br>
> Wonjong Jang, Yucheol Jung, Gyeongmin Lee, Seungyong Lee

## Update log

March 17, 2026: Add interactive Polyscope visualization (`--vis` flag)
August 13, 2025: Release initial version

## Requirements

- Python 3.x
- PyTorch (with CUDA)
- [potpourri3d](https://github.com/nmwsharp/potpourri3d)
- [polyscope](https://polyscope.run/py/) *(required only for `--vis` mode)*
- [largesteps](https://github.com/rgl-epfl/large-steps-pytorch)
- [torch-mesh-isect](https://github.com/vchoutas/torch-mesh-isect) *(included as submodule under `externals/`)*

## Usage

### Configuration

Experiments are configured via YAML files. Example (`configs/smpl_1.yaml`):

```yaml
expname: SMPL
objpath: ./collisions/SMPL_Ap.obj
optimizer: MomentumBrake
lr: 0.001
savepath: ./results
max_collisions: 8
energy: signed_TPE_verts
constraints:
  - curvature
```

**Config fields:**

| Field | Options | Description |
|---|---|---|
| `expname` | any string | Experiment name used for output filenames |
| `objpath` | path | Input mesh (`.obj`), triangles or quads |
| `savepath` | path | Directory for output meshes |
| `max_collisions` | int | Max BVH collision pairs to detect per query |
| `lap` | `cotan`, `curv` | Laplacian type for parameterization |
| `optimizer` | `Adam`, `GD`, `MomentumBrake`, `AdamUniform` | Optimizer |
| `lr` | float | Learning rate |
| `energy` | `signed_TPE`, `signed_TPE_verts`, `TPE`, `p2plane`, `conical` | Penetration energy |
| `constraints` | `volume`, `area`, `curvature` | Geometric regularization terms |

### Running repair

```bash
# Headless (saves intermediate .obj files every 10 steps + best/final)
python repair_factory.py --config configs/smpl_1.yaml

# Interactive visualization
python repair_factory.py --config configs/smpl_1.yaml --vis
```

### Interactive visualization (`--vis`)

Passing `--vis` opens a [Polyscope](https://polyscope.run/py/) window showing the mesh as it is optimized.

**Controls:**

| Button | Behavior |
|---|---|
| **Step** | Execute one optimization iteration |
| **Run** | Auto-play all remaining iterations (one per frame) |
| **Pause** | Stop auto-play (appears in place of Run while running) |

A per-face scalar overlay (`collisions`, red colormap) highlights which faces are involved in intersections. The intensity reflects the number of collision pairs involving each face.

The stats panel shows the current step, collision count, penetration loss, regularization loss, and the best solution found so far.

Best and final meshes are saved to `savepath` when all steps complete or when the window is closed after completion.

### Output files

| File | Description |
|---|---|
| `<expname>_init.obj` | Normalized input mesh (headless only) |
| `<expname>_010.obj` … | Intermediate meshes every 10 steps (headless only) |
| `<expname>_best.obj` | Mesh at the iteration with fewest collisions |
| `<expname>_final.obj` | Mesh at the last iteration |

## Benchmark data

Please contact wonjong@postech.ac.kr to request our benchmark data
