import numpy as np
from numba import cuda, float32
import math


@cuda.jit(device=True)
def device_calculate_heuristic_f(path_cost, mid_velocity, velocity, dist_to_obs, weights, cur_coordinates, goal):
    c1 = weights[0]
    c2 = weights[1]
    # c3 = weights[2] 
    mv = float32(mid_velocity)
    eps = float32(1e-3)
    dx = float32(cur_coordinates[0] - goal[0])
    dy = float32(cur_coordinates[1] - goal[1])
    dz = float32(cur_coordinates[2] - goal[2])
    f2Goal = math.sqrt(dx*dx + dy*dy + dz*dz)
    return path_cost + c1 * f2Goal/mv + c2 * 1/(dist_to_obs + eps)

@cuda.jit
def kernal_calculate_heuristic_f(path_cost, mid_velocity, velocity, dist_to_obs, weights, cur_coordinates , goal, heuristic):
    tid = cuda.grid(1)
    if tid < heuristic.shape[0]:
        heuristic[tid] = device_calculate_heuristic_f(path_cost[tid], mid_velocity, velocity[tid], dist_to_obs[tid], weights[tid],
                                             cur_coordinates[tid], goal)
        
def calculate_heuristic_init_f(path_cost, mid_velocity, velocity, dist_to_obs, weights, cur_coordinates , goal):
    threads_per_block = 128
    blocks_per_grid = (path_cost.shape[0] + (threads_per_block - 1)) // threads_per_block
    heuristic = cuda.device_array(path_cost.shape[0], dtype = path_cost.dtype)
    kernal_calculate_heuristic_f[blocks_per_grid, threads_per_block](path_cost, mid_velocity, velocity, dist_to_obs,
                                                                   weights, cur_coordinates, goal, heuristic)
    cuda.synchronize()
    return heuristic.copy_to_host()