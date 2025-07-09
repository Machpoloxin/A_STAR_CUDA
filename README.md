# A_STAR_CUDA

A high-performance CUDA implementation of the A* pathfinding algorithm in 3D voxel grids. This repository provides a GPU-accelerated A* variant, leveraging Numba/CUDA for efficient neighbor expansion, heuristic computation, and priority queue management.

## Features

- **Parallel Neighbor Expansion**: Expands neighbors in parallel across threads for fast processing.
- **Custom Heuristic**: Supports voxel-to-obstacle distance heuristic using precomputed distance fields.
- **GPU Priority Queue**: Implements a binary heap (three-way) for frontier management entirely on device.
- **Configurable Weights & Velocities**: Incorporates velocity and weight parameters for dynamic cost calculations.
- **Modular Design**: Separate modules for grid management, distance field computation, and A* search.

## Table of Contents

- [A\_STAR\_CUDA](#a_star_cuda)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [File Structure](#file-structure)
  - [Performance](#performance)
    - [Test on an NVIDIA RTX 4060:](#test-on-an-nvidia-rtx-4060)
  - [Contributing](#contributing)
  - [License](#license)

## Requirements

- Ubuntu 22.04+
- Python 3.8+
- Numba
- CUDA Toolkit (Compute Capability 12.0+ recommended)
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:Machpoloxin/A_STAR_CUDA.git  
   cd A_STAR_CUDA
   ```
2. Create a Python virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Ensure CUDA is properly installed and accessible.

## Usage
   ```python
   from a_star_cuda import run_astar_search_cuda
   import numpy as np

   # Define a 3D grid: 0-free, 1-occupied
   grid = np.zeros((depth, height, width), dtype=np.int32)
   # ... set obstacles ...

   # Precompute distance field, velocity grid, etc.
   distance_field = compute_distance_field(grid)
   velocity_grid = np.full_like(grid, initial_velocity, dtype=np.float32)

   # Define start and goal
   start = (x0, y0, z0)
   goal = (xg, yg, zg)

   # Run A* on GPU
   paths, lengths = run_astar_search_cuda(
                     d_grid, d_start, d_start_v, 
                     d_goal, d_goal_v, 
                     d_weights, 
                     d_velocity_grid, 
                     d_distance_field, 400, max_time = 10000)
   ```
For more details, refer to **examples/test.ipynb**

## File Structure
```
A_STAR_CUDA/
├── aStarCuda/
│   ├── __init__.py
│   ├── expander.py       
│   ├── globalEnv.py   
│   ├── heuristic.py       
│   ├── reconstructor.py
|   └── runner.py           # Main loop of A*
├── heapMangager/
│   ├── __init__.py
│   ├── heapManager.py
│   └── benchmark.py
├── obstacle_interface/
│   ├── __init__.py
│   └── obs_interface.py
├── examples/
│   └── test.ipynb          # Usage
├── requirements.txt
└── README.md               # This file
```

## Performance
### Test on an NVIDIA RTX 4060:

| Grid Size    |  Threads |         CPU A* (ms)         |     GPU A* (ms)    |
|:------------:|:--------:|:---------------------------:|:------------------:|
| 50×50×50     |    5     |             20              |     around 1500    |
| 50×50×50     |   400    |         around 1500         |     around 1500    |


## Contributing
Contributions are welcome! Please open issues or submit pull requests.

## License
This project is licensed under the "---" License. See **LICENSE** for details.
```
```