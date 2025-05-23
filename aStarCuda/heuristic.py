import numpy as np
from numba import njit, prange, cuda, float32, int32
import math
import time
import numba


# __device__
@cuda.jit(device=True)
def device_calculate_cost2go_g(path_cost, step_cost ,cur_velocity, dist_to_obs, weights):
    '''
    Because function compiled with njit cannot use cuda.jit device functions.
    :param path_cost: The cost of the path so far
    :param cur_velocity: The velocity at the current node
    :param dist_to_obs: The distance to the closest obstacle
    :param weights: The weights for the heuristic function
    :return: The cost2go: g() value
    '''
    c1 = weights[0]
    return path_cost + step_cost/(cur_velocity+1) * c1

@cuda.jit
def kernal_calculate_cost2go_g(path_new_cost, path_cost, step_cost ,cur_velocity, dist_to_obs, weights):
    '''
    The kernel function to calculate the heuristic value for each node.
    :param path_new_cost: The cost of the path so far + the cost to go
    :param path_cost: The cost of the path so far
    :param cur_velocity: The velocity at the current node
    :param dist_to_obs: The distance to the closest obstacle
    :param weights: The weights for the heuristic function
    :return: The cost2go: g() value
    '''
    tid = cuda.grid(1)
    if tid < weights.shape[0]:
        path_new_cost[tid] = device_calculate_cost2go_g(path_cost[tid], step_cost[tid], cur_velocity[tid], dist_to_obs[tid], weights[tid])
        
def calculate_cost2go_g(path_cost, step_cost ,cur_velocity, dist_to_obs, weights):
    '''
    The function to calculate the heuristic value for each node.
    :param path_cost: The cost of the path so far
    :param mid_velocity: The velocity at the middle of the path
    :param velocity: The velocity at the current node
    :param dist_to_obs: The distance to the closest obstacle
    :param weights: The weights for the heuristic function
    :param cur_coordinates: The x, y, z-coordinate of the current node
    :param goal: The x, y, z -coordinate of the goal node
    :return: The heuristic value for each node
    '''
    threads_per_block = 128
    blocks_per_grid = (weights.shape[0] + (threads_per_block - 1)) // threads_per_block # round up
    path_new_cost = cuda.device_array(weights.shape[0], dtype = path_cost.dtype)
    kernal_calculate_cost2go_g[blocks_per_grid, threads_per_block](path_new_cost, path_cost,  step_cost ,cur_velocity, dist_to_obs, weights)
    cuda.synchronize()
    return path_new_cost.copy_to_host()



# __device__
@cuda.jit(device=True)
def device_calculate_heuristic_f(path_cost, mid_velocity, velocity, dist_to_obs, weights, cur_coordinates, goal):
    '''
    Because function compiled with njit cannot use cuda.jit device functions.
    :param path_cost: The cost of the path so far
    :param mid_velocity: The velocity at the middle of the path
    :param velocity: The velocity at the current node
    :param dist_to_obs: The distance to the closest obstacle
    :param weights: The weights for the heuristic function
    :param cur_coordinates : The x, y, z-coordinate of the current node
    :param goal: The x, y, z -coordinate of the goal node
    :return: The heuristic value
    '''
    c1 = weights[0]
    c2 = weights[1]
    c3 = weights[2]
    mv = float32(mid_velocity)
    eps = float32(1e-3)
    dx = float32(cur_coordinates[0] - goal[0])
    dy = float32(cur_coordinates[1] - goal[1])
    dz = float32(cur_coordinates[2] - goal[2])
    f2Goal = math.sqrt(dx*dx + dy*dy + dz*dz)
    return path_cost + c1 * f2Goal/mv + c2 * 1/(dist_to_obs + eps) 


@cuda.jit
def kernal_calculate_heuristic_f(path_cost, mid_velocity, velocity, dist_to_obs, weights, cur_coordinates , goal, heuristic):
    '''
    The kernel function to calculate the heuristic value for each node.
    :param path_cost: The cost of the path so far
    :param mid_velocity: The velocity at the middle of the path
    :param velocity: The velocity at the current node
    :param dist_to_obs: The distance to the closest obstacle
    :param weights: The weights for the heuristic function
    :param cur_coordinates: The x, y, z-coordinate of the current node
    :param goal: The x, y, z -coordinate of the goal node
    :param heuristic: The heuristic value for each node
    '''
    tid = cuda.grid(1)
    if tid < heuristic.shape[0]:
        heuristic[tid] = device_calculate_heuristic_f(path_cost[tid], mid_velocity, velocity[tid], dist_to_obs[tid], weights[tid],
                                             cur_coordinates[tid], goal)
        
def calculate_heuristic_init_f(path_cost, mid_velocity, velocity, dist_to_obs, weights, cur_coordinates , goal):
    '''
    The function to calculate the heuristic value for each node.
    :param path_cost: The cost of the path so far
    :param mid_velocity: The velocity at the middle of the path
    :param velocity: The velocity at the current node
    :param dist_to_obs: The distance to the closest obstacle
    :param weights: The weights for the heuristic function
    :param cur_coordinates: The x, y, z-coordinate of the current node
    :param goal: The x, y, z -coordinate of the goal node
    :return: The heuristic value for each node
    '''
    threads_per_block = 128
    blocks_per_grid = (path_cost.shape[0] + (threads_per_block - 1)) // threads_per_block # round up
    heuristic = cuda.device_array(path_cost.shape[0], dtype = path_cost.dtype)
    kernal_calculate_heuristic_f[blocks_per_grid, threads_per_block](path_cost, mid_velocity, velocity, dist_to_obs,
                                                                   weights, cur_coordinates, goal, heuristic)
    cuda.synchronize()
    return heuristic.copy_to_host()

