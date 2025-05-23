import numpy as np
from numba import njit, prange, cuda, float32, int32
import math
import time
import numba

from aStarCuda.globalEnv import LOCAL_PATH_LENGTH

@cuda.jit(device=True)
def device_reconstruct_path(pred_x, pred_y, pred_z, pred_v,
                            start, start_v, goal, goal_v,
                            path_buf, path_len_buf, tid, max_steps = 200):
    ''' 
    Reconstruct the path from the predicted path.
    :param pred_... : The predecessor coordinates and velocity
    :param start: The start coordinates
    :param start_v: The velocity at the start node
    :param goal: The goal coordinates
    :param goal_v: The velocity at the goal node
    '''
    D, H, W = pred_x.shape
    #max_steps = D*H*W
    local_path = cuda.local.array((LOCAL_PATH_LENGTH,), dtype=int32)  # local array for all path
    local_v    = cuda.local.array((LOCAL_PATH_LENGTH,), dtype=float32)
    idx = 0
    cx, cy, cz = goal[0], goal[1], goal[2]
    local_path[idx] = (cx*H + cy)*W + cz  # coding the idx(linear)
    local_v   [idx] = goal_v

    # backtracking
    while not (cx == start[0] and cy == start[1] and cz == start[2]):
        if idx + 1 >= max_steps:
            path_len_buf[tid] = 0
            return
        px = pred_x[cz, cy, cx]
        py = pred_y[cz, cy, cx]
        pz = pred_z[cz, cy, cx]

        if px < 0 or py < 0 or pz < 0 or px >= W or py >= H or pz >= D:
            path_len_buf[tid] = 0  
            return
        
        idx += 1
        cx, cy, cz = px, py, pz
        local_path[idx] = (cx*H + cy)*W + cz
        local_v   [idx] = (start_v if (cx==start[0] and cy==start[1] and cz==start[2])
                            else pred_v[cz, cy, cx])
    # Flip the path
    for i in range((idx+1)//2):
        j = idx - i
        # swap path
        t = local_path[i]; local_path[i] = local_path[j]; local_path[j] = t
        # swap v
        tv = local_v[i]; local_v[i] = local_v[j]; local_v[j] = tv

    # write back to global buffer
    for i in range(idx+1):
        linear = local_path[i]
        x = linear // (H*W)
        y = (linear // W) % H
        z = linear % W
        path_buf[tid, i, 0] = x
        path_buf[tid, i, 1] = y
        path_buf[tid, i, 2] = z
        path_buf[tid, i, 3] = local_v[i]

    if idx + 1 >= max_steps:
        path_len_buf[tid] = 0
    else:
        path_len_buf[tid] = idx + 1
        
        
        
        
@cuda.jit
def kernel_reconstruct_if_goal(
    pred_x, pred_y, pred_z, pred_v,
    starts, starts_v, 
    goal, goals_v,
    path_buf, path_len_buf,  d_finished_markers, L_max
):
    tid = cuda.grid(1)
    if tid >= pred_x.shape[0] or d_finished_markers[tid] != 1:
        return
    
    if tid < path_len_buf.shape[0]:
        path_len_buf[tid] = 0


    device_reconstruct_path(
            pred_x[tid], pred_y[tid], pred_z[tid], pred_v[tid],
            starts, starts_v, goal, goals_v,
            path_buf, path_len_buf, tid, L_max
       )
        
def reconstruct_path_if_goal (
                    pred_x, pred_y, pred_z, pred_v,
                    starts, starts_v, 
                    goals, goals_v,
                    d_finished_markers, L_max = 200):
    '''
    Reconstruct the path from the predicted path.
    :param pred_... : The predecessor coordinates and velocity
    :param starts: The start coordinates
    :param starts_v: The velocity at the start node
    :param goals: The goal coordinates
    :param goals_v: The velocity at the goal node
    '''

    num_threads = pred_x.shape[0]
    d_paths     = cuda.device_array((num_threads, L_max, 4), dtype=np.float32)
    d_path_lens = cuda.device_array(num_threads, dtype=np.int32)
    threads_per_block = 128
    blocks_per_grid = (num_threads + (threads_per_block - 1)) // threads_per_block # round up
    kernel_reconstruct_if_goal[blocks_per_grid, threads_per_block](
        pred_x, pred_y, pred_z, pred_v,
        starts, starts_v, 
        goals, goals_v,
        d_paths, d_path_lens, d_finished_markers, L_max
    )
    cuda.synchronize()
    return d_paths.copy_to_host(), d_path_lens.copy_to_host()
        
        
        

