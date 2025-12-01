import numpy as np
from numba import cuda, float32, int32
from aStarCuda.globalEnv import LOCAL_PATH_LENGTH

@cuda.jit(device=True)
def device_reconstruct_path(
    pred_action,
    start, start_v, goal, goal_v,
    velocity_grid,
    path_buf, path_len_buf, tid, max_steps
):
    # Layout: (D, H, W, N)
    D, H, W, N = pred_action.shape
    
    idx = 0
    cx, cy, cz = goal[0], goal[1], goal[2]
    # From goal to start
    path_buf[tid, idx, 0] = float32(cx)
    path_buf[tid, idx, 1] = float32(cy)
    path_buf[tid, idx, 2] = float32(cz)
    path_buf[tid, idx, 3] = float32(goal_v)

    while not (cx == start[0] and cy == start[1] and cz == start[2]):
        if idx + 1 >= max_steps:
            path_len_buf[tid] = 0  
            return
        action_code = pred_action[cz, cy, cx, tid]
        
        if action_code == -1:
            path_len_buf[tid] = 0
            return
        move_idx = action_code % 27
        vi = action_code // 27
    
        dz = (move_idx // 9) - 1
        rem = move_idx % 9
        dy = (rem // 3) - 1
        dx = (rem % 3) - 1
        
        px = cx - dx
        py = cy - dy
        pz = cz - dz
        
        # Bounds check
        if px < 0 or px >= W or py < 0 or py >= H or pz < 0 or pz >= D:
            path_len_buf[tid] = 0
            return

        p_vel = velocity_grid[vi]

        idx += 1
        cx, cy, cz = px, py, pz
        
        path_buf[tid, idx, 0] = float32(cx)
        path_buf[tid, idx, 1] = float32(cy)
        path_buf[tid, idx, 2] = float32(cz)
        path_buf[tid, idx, 3] = float32(p_vel)

    count = idx + 1

    for i in range(count // 2):
        j = count - 1 - i
        
        xi, yi, zi, vi = path_buf[tid, i, 0], path_buf[tid, i, 1], path_buf[tid, i, 2], path_buf[tid, i, 3]
        xj, yj, zj, vj = path_buf[tid, j, 0], path_buf[tid, j, 1], path_buf[tid, j, 2], path_buf[tid, j, 3]
        # Write i <- j
        path_buf[tid, i, 0] = xj
        path_buf[tid, i, 1] = yj
        path_buf[tid, i, 2] = zj
        path_buf[tid, i, 3] = vj
        # Write j <- i
        path_buf[tid, j, 0] = xi
        path_buf[tid, j, 1] = yi
        path_buf[tid, j, 2] = zi
        path_buf[tid, j, 3] = vi

    path_len_buf[tid] = count

@cuda.jit
def kernel_reconstruct_wrapper(
    pred_action,
    start, start_v, 
    goal, goal_v,
    velocity_grid,
    d_paths, d_path_lens, d_finished_markers, L_max
):
    tid = cuda.grid(1)
    if tid >= d_finished_markers.shape[0]:
        return
        
    if d_finished_markers[tid] != 1:
        d_path_lens[tid] = 0
        return

    device_reconstruct_path(
        pred_action,
        start, start_v, goal, goal_v,
        velocity_grid,
        d_paths, d_path_lens, tid, L_max
    )

def reconstruct_path_if_goal(
    d_pred_action,
    start, start_v, 
    goal, goal_v,
    d_velocity_grid,
    d_finished_markers, L_max=200
):
    num_threads = d_finished_markers.shape[0]
    
    d_paths = cuda.device_array((num_threads, L_max, 4), dtype=np.float32)
    d_path_lens = cuda.device_array(num_threads, dtype=np.int32)
    
    threads_per_block = 128
    blocks = (num_threads + threads_per_block - 1) // threads_per_block
    
    d_start = cuda.to_device(start)
    d_goal = cuda.to_device(goal)
    
    kernel_reconstruct_wrapper[blocks, threads_per_block](
        d_pred_action,
        d_start, start_v, 
        d_goal, goal_v,
        d_velocity_grid,
        d_paths, d_path_lens, d_finished_markers, L_max
    )
    cuda.synchronize()
    return d_paths.copy_to_host(), d_path_lens.copy_to_host()