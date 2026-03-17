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

### Quick start with toy data

A set of self-intersecting toy meshes is included under `data/misc/` along with a ready-to-use config:

| Mesh | File |
|---|---|
| Celtic knot | `celtic_knot.obj` |
| Cinquefoil knot | `cinquefoil_knot.obj` |
| Disc Klein bottle | `disc_kleinbottle.obj` |
| Figure-of-eight knot | `figure_of_eight_knot.obj` |
| Knot 8-18 | `knot_8_18.obj` |
| Möbius strip | `mobius_strip.obj` |
| Pretzel | `pretzel.obj` |
| Septoil knot | `septoil_knot.obj` |
| Three-twist knot | `three_twist_knot.obj` |
| Trefoil knot | `trefoil_knot.obj` |

Run the Celtic knot example (default in `configs/misc.yaml`):

```bash
# Headless
python repair_factory.py --config configs/misc.yaml

# With interactive visualization
python repair_factory.py --config configs/misc.yaml --vis
```

To try a different mesh, edit the `objpath` field in `configs/misc.yaml`.

### Configuration

Experiments are configured via YAML files. Example (`configs/misc.yaml`):

```yaml
expname: misc
objpath: ./data/misc/celtic_knot.obj
optimizer: GD
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
| `optimizer` | `Adam`, `GD`, `MomentumBrake`, `AdamUniform` | Optimizer |
| `lr` | float | Learning rate |
| `energy` | `signed_TPE`, `signed_TPE_verts`, `TPE`, `p2plane`, `conical` | Penetration energy |
| `constraints` | `volume`, `area`, `curvature` | Geometric regularization terms |

### Running repair

```bash
# Headless (saves intermediate .obj files every 10 steps + best/final)
python repair_factory.py --config configs/your_config.yaml

# Interactive visualization
python repair_factory.py --config configs/your_config.yaml --vis
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
