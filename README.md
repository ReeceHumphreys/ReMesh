# ReMesh

ReMesh is a Python tool for region-based refinement of 3D mesh models. It supports targeted mesh operations (subdivision and simplification) on specified regions of a mesh.

## Features

- Load 3D meshes directly from an STL file.
- Region-specific mesh subdivision with optional spherical interpolation for curvature preservation.
- Region-specific mesh simplification (quadratic decimation) for face reduction.
- Visualize before/after refinement with color-coded regions.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

1. Install Poetry (if not already installed):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:

   ```bash
   poetry install
   ```

## Usage

Run the `main.py` script with the input STL file and a YAML configuration file defining the refinement regions:

```bash
poetry run python main.py <path/to/input.stl> <path/to/config.yaml> [--out <output.stl>] [--debug]
```

### Arguments

- `<path/to/input.stl>`: Path to the input STL file.
- `<path/to/config.yaml>`: Path to the YAML config file defining the regions for refinement.
- `--out`: (Optional) Path for the refined mesh output (default: `refined.stl`).
- `--debug`: (Optional) Show before/after colored visualizations of the specified regions.

### Example

```bash
poetry run python main.py ../IXV/models/Coarse/lf_0_rf_0-fixed-normals.stl test.yaml --out refined.stl --debug
```

## Configuration File

The YAML configuration file defines the regions for refinement. Each region specifies the bounds and the operation to perform (e.g., subdivision or combination).

```yaml
regions:
  - name: <region_name>
    bounds:
      x: [min, max]
      y: [min, max]
      z: [min, max]
    operation: subdivide
    max_edge_length: <float>
    use_spherical_interp: true
    curvature_center: [x, y, z]

  - name: <other_region>
    bounds:
      x: [min, max]
      y: [min, max]
      z: [min, max]
    operation: combine
    reduction_ratio: <float between 0 and 1>
```

## Output

The refined mesh is saved to the specified output STL file. If `--debug` is enabled, a visualization of the mesh before and after refinement is displayed.
