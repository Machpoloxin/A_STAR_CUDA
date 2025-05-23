import numpy as np
from numba import njit, prange, cuda, float32, int32
import math
import time
import numba

from heapManager.heapManger import MyHeapManager
from aStarCuda.heuristic import calculate_heuristic_init_f
from aStarCuda.expander import expand_neighbors
from aStarCuda.reconstructor import reconstruct_path_if_goal
from aStarCuda.globalEnv import MAX_NEI, INF



def run_astar_search_numba(grid, start, start_v, goal, goal_v, weights, velocity_grid, distance_field, stop_threshold ,max_time = 1e10):
    parallel_parameter = weights.shape[0]
    depth, height, width = grid.shape

    # Host initialization 
    h_cost = np.full((parallel_parameter, depth, height, width), INF, dtype=np.float32)
    h_cost[:, start[2], start[1], start[0]] = 0.0 
    h_init_cost = np.full((parallel_parameter), 0, dtype=np.float32)
    h_pred_x = -np.ones((parallel_parameter, depth, height, width), dtype=np.int32)
    h_pred_y = -np.ones((parallel_parameter, depth, height, width), dtype=np.int32)
    h_pred_z = -np.ones((parallel_parameter, depth, height, width), dtype=np.int32)
    h_pred_v = -np.ones((parallel_parameter, depth, height, width), dtype=np.float32) 
    n_cells = depth * height * width
    h_paths = np.zeros((parallel_parameter, 2 * n_cells, 4), dtype=np.float32)
    h_path_lens = np.zeros(parallel_parameter, dtype=np.int32)
    h_mid_velocity = np.float32((velocity_grid[0] + velocity_grid[-1]) / 2) #running time (const)
    h_init_velocity = np.zeros((parallel_parameter,), dtype=np.float32)
    single_dist = np.float32(distance_field[start[2], start[1], start[0]])
    h_init_dist = np.full((parallel_parameter,), single_dist, dtype=np.float32)
    h_finished_markers = np.zeros((parallel_parameter,), dtype=np.int32)

    # Device initialization
    fullBinThree = MyHeapManager(parallel_parameter, n_cells) 
    d_start = cuda.to_device(start)
    h_start_coords = np.tile(start, (parallel_parameter,1)).astype(np.int32)   # (N,3)
    d_start_coords = cuda.to_device(h_start_coords)
    d_start_v = np.float32(start_v)
    d_goal = cuda.to_device(goal)   
    d_goal_v = np.float32(goal_v)
    d_weights = cuda.to_device(weights)
    d_velocity_grid = cuda.to_device(velocity_grid)
    #d_mid_velocity = cuda.to_device(h_mid_velocity)
    d_init_velocity = cuda.to_device(h_init_velocity)
    d_distance_field = cuda.to_device(distance_field)
    d_cost = cuda.to_device(h_cost)
    d_init_cost = cuda.to_device(h_init_cost)
    d_pred_x = cuda.to_device(h_pred_x)
    d_pred_y = cuda.to_device(h_pred_y)
    d_pred_z = cuda.to_device(h_pred_z)
    d_pred_v = cuda.to_device(h_pred_v)
    #d_paths = cuda.to_device(h_paths)
    #d_path_lens = cuda.to_device(h_path_lens)
    d_init_dist = cuda.to_device(h_init_dist)
    d_finished_markers = cuda.to_device(h_finished_markers)


    initial_priority = calculate_heuristic_init_f(d_init_cost, h_mid_velocity, d_init_velocity, 
                                          d_init_dist, d_weights, d_start_coords , d_goal)
    
    h_elems = np.empty((parallel_parameter, 4), dtype=np.float32)
    h_elems[:, 0] = initial_priority        
    h_elems[:, 1] = start[0]                # x
    h_elems[:, 2] = start[1]                # y 
    h_elems[:, 3] = start[2]
    d_elems = cuda.to_device(h_elems)
    fullBinThree.push(d_elems)
    del d_elems

    timerAstar = 0 #running time: ms
    
    while timerAstar < max_time:
        t0 = time.perf_counter()
        # Pop the element from the heap
        popped_elements = fullBinThree.pop()
        cur_coordinates = popped_elements[:, 1:4].astype(np.int32) # x, y, z
        d_cur_coordinates = cuda.to_device(cur_coordinates)

        fullBinThree = expand_neighbors(fullBinThree, d_cur_coordinates, d_goal,
                                        grid, d_cost, d_pred_x, d_pred_y, d_pred_z, d_pred_v,
                                        d_distance_field, d_weights, h_mid_velocity, d_velocity_grid,
                                        width, height, depth, d_finished_markers, MAX_NEI)
        
        h_finished_markers = d_finished_markers.copy_to_host()

        if np.count_nonzero(h_finished_markers == 1) >= stop_threshold:
            print("find enough paths")
            break

        t1 = time.perf_counter()
        timerAstar += (t1 - t0) * 1000

    h_paths, h_path_lens = reconstruct_path_if_goal (d_pred_x, d_pred_y, d_pred_z, d_pred_v,
                                                        d_start, d_start_v, d_goal, d_goal_v,
                                                        d_finished_markers, n_cells)
        

    return h_paths, h_path_lens
    

