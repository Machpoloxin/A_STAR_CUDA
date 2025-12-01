import numpy as np
from numba import cuda
import time

from heapManager.heapManger import MyHeapManager
from aStarCuda.heuristic import calculate_heuristic_init_f
from aStarCuda.expander import expand_neighbors_fused
from aStarCuda.reconstructor import reconstruct_path_if_goal
from aStarCuda.globalEnv import INF

def run_astar_search_numba(grid, start, start_v, goal, goal_v, weights, velocity_grid, distance_field, stop_threshold, max_time=1e10):
    parallel_parameter = weights.shape[0]
    depth, height, width = grid.shape

    # Memory Layout Optimization: (D, H, W, N)
    h_cost = np.full((depth, height, width, parallel_parameter), INF, dtype=np.float32)
    
    # Initialize Start Cost
    h_cost[start[2], start[1], start[0], :] = 0.0 
    
    # -1 means no parent
    h_pred_action = np.full((depth, height, width, parallel_parameter), -1, dtype=np.int8)

    # Heuristic Helpers
    h_mid_velocity = np.float32((velocity_grid[0] + velocity_grid[-1]) / 2) 
    h_init_velocity = np.zeros((parallel_parameter,), dtype=np.float32)
    single_dist = np.float32(distance_field[start[2], start[1], start[0]])
    h_init_dist = np.full((parallel_parameter,), single_dist, dtype=np.float32)
    h_finished_markers = np.zeros((parallel_parameter,), dtype=np.int32)
    
    h_init_cost = np.zeros((parallel_parameter), dtype=np.float32)

    # --- Device Allocation ---
    n_cells = depth * height * width
    fullBinThree = MyHeapManager(parallel_parameter, n_cells) 
    
    h_start_coords = np.tile(start, (parallel_parameter,1)).astype(np.int32)
    d_start_coords = cuda.to_device(h_start_coords)
    d_goal = cuda.to_device(goal)   
    
    d_weights = cuda.to_device(weights)
    d_velocity_grid = cuda.to_device(velocity_grid)
    d_init_velocity = cuda.to_device(h_init_velocity)
    d_distance_field = cuda.to_device(distance_field)

    d_cost = cuda.to_device(h_cost)
    d_pred_action = cuda.to_device(h_pred_action)
    d_finished_markers = cuda.to_device(h_finished_markers)
    d_init_dist = cuda.to_device(h_init_dist)

    # Initial Priority Calculation
    initial_priority = calculate_heuristic_init_f(
        cuda.to_device(h_init_cost), h_mid_velocity, d_init_velocity, 
        d_init_dist, d_weights, d_start_coords, d_goal
    )
    
    h_elems = np.empty((parallel_parameter, 4), dtype=np.float32)
    h_elems[:, 0] = initial_priority        
    h_elems[:, 1] = start[0]
    h_elems[:, 2] = start[1] 
    h_elems[:, 3] = start[2]
    d_elems = cuda.to_device(h_elems)
    fullBinThree.push(d_elems)
    
    timerAstar = 0
    
    # --- Main Loop ---
    while timerAstar < max_time:
        t0 = time.perf_counter()
        
        # Single Kernel Launch
        expand_neighbors_fused(
            fullBinThree, 
            grid, d_cost, d_pred_action,
            d_distance_field, d_weights, h_mid_velocity, d_velocity_grid,
            width, height, depth, d_finished_markers, d_goal
        )
        
        h_finished_markers = d_finished_markers.copy_to_host()

        if np.count_nonzero(h_finished_markers == 1) >= stop_threshold:
            print(f"Find enough paths: {np.count_nonzero(h_finished_markers == 1)}")
            break
        
        t1 = time.perf_counter()
        timerAstar += (t1 - t0) * 1000

    # Path Reconstruction 
    h_paths, h_path_lens = reconstruct_path_if_goal(
        d_pred_action,
        start, start_v, goal, goal_v,
        d_velocity_grid,
        d_finished_markers, 
        L_max=200
    )

    return h_paths, h_path_lens