import numpy as np
from numba import cuda, float32, int32, int8
import math


@cuda.jit(device=True)
def device_heap_push(heap, heap_size, elem):
    idx = heap_size[0]
    heap[idx, 0] = elem[0]
    heap[idx, 1] = elem[1]
    heap[idx, 2] = elem[2]
    heap[idx, 3] = elem[3]
    heap_size[0] += 1
    
    i = idx
    while i > 0:
        parent = (i - 1) // 2
        if heap[parent, 0] > heap[i, 0]:
            # Swap
            t0, t1, t2, t3 = heap[parent, 0], heap[parent, 1], heap[parent, 2], heap[parent, 3]
            heap[parent, 0] = heap[i, 0]
            heap[parent, 1] = heap[i, 1]
            heap[parent, 2] = heap[i, 2]
            heap[parent, 3] = heap[i, 3]
            heap[i, 0] = t0
            heap[i, 1] = t1
            heap[i, 2] = t2
            heap[i, 3] = t3
            i = parent
        else:
            break

@cuda.jit
def kernel_expand_all_neighbors(
    heap, heap_size,
    grid, cost, pred_action,
    distance_field, weights, mid_velocity, velocity_grid,
    width, height, depth, 
    d_finished_markers,
    goal_coords
):
    tid = cuda.grid(1)
    
    # cost shape is (D, H, W, N) -> tid is the last dimension
    if tid >= cost.shape[3] or d_finished_markers[tid] == 1:
        return

    if heap_size[tid] == 0:
        return 

    # --- Pop Top Element ---
    curr_f = heap[tid, 0, 0]
    cx = int32(heap[tid, 0, 1])
    cy = int32(heap[tid, 0, 2])
    cz = int32(heap[tid, 0, 3])
    
    # Check Goal
    if cx == goal_coords[0] and cy == goal_coords[1] and cz == goal_coords[2]:
        d_finished_markers[tid] = 1
        return

    # Heap Remove & Bubble Down
    last_idx = heap_size[tid] - 1
    heap[tid, 0, 0] = heap[tid, last_idx, 0]
    heap[tid, 0, 1] = heap[tid, last_idx, 1]
    heap[tid, 0, 2] = heap[tid, last_idx, 2]
    heap[tid, 0, 3] = heap[tid, last_idx, 3]
    heap_size[tid] = last_idx
    
    i = 0
    size = heap_size[tid]
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < size and heap[tid, left, 0] < heap[tid, smallest, 0]:
            smallest = left
        if right < size and heap[tid, right, 0] < heap[tid, smallest, 0]:
            smallest = right
        if smallest != i:
            t0, t1, t2, t3 = heap[tid, i, 0], heap[tid, i, 1], heap[tid, i, 2], heap[tid, i, 3]
            heap[tid, i, 0] = heap[tid, smallest, 0]
            heap[tid, i, 1] = heap[tid, smallest, 1]
            heap[tid, i, 2] = heap[tid, smallest, 2]
            heap[tid, i, 3] = heap[tid, smallest, 3]
            heap[tid, smallest, 0] = t0
            heap[tid, smallest, 1] = t1
            heap[tid, smallest, 2] = t2
            heap[tid, smallest, 3] = t3
            i = smallest
        else:
            break
            
    current_cost = cost[cz, cy, cx, tid]
    
    # Iterate 27 directions
    for move_idx in range(27):
        # Decode move_idx (0..26) -> dx, dy, dz
        dz = (move_idx // 9) - 1
        rem = move_idx % 9
        dy = (rem // 3) - 1
        dx = (rem % 3) - 1

        if dx == 0 and dy == 0 and dz == 0:
            continue

        nx = cx + dx
        ny = cy + dy
        nz = cz + dz

        # Boundary Check
        if nx < 0 or nx >= width or ny < 0 or ny >= height or nz < 0 or nz >= depth:
            continue
        
        # Obstacle Check
        if grid[nz, ny, nx] != 0:
            continue

        # Move Cost
        dist_sq = dx*dx + dy*dy + dz*dz
        move_cost = math.sqrt(float32(dist_sq))

        # Velocity Loop
        for vi in range(velocity_grid.shape[0]):
            vel = velocity_grid[vi]
            # Heuristic Weight [0]: c1
            added_cost = (move_cost / (vel + 1.0)) * weights[tid, 0]
            new_g = current_cost + added_cost
            
            # Read old cost from Global Memory 
            old_g = cost[nz, ny, nx, tid]
            if new_g < old_g:
                cost[nz, ny, nx, tid] = new_g
                # Calculate Heuristic Inline
                dist_obs = distance_field[nz, ny, nx]
                c1 = weights[tid, 0]
                c2 = weights[tid, 1]
                
                diff_x = nx - goal_coords[0]
                diff_y = ny - goal_coords[1]
                diff_z = nz - goal_coords[2]
                euc = math.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
                
                h_val = c1 * (euc / mid_velocity) + c2 * (1.0 / (dist_obs + 1e-3))
                f_val = new_g + h_val

                # Store Action (Compressed Predecessor)
                # action_code = move_idx (0-26) + 27 * velocity_index
                action_code = int8(move_idx + 27 * vi)
                pred_action[nz, ny, nx, tid] = action_code

                # Push to Heap
                new_item = cuda.local.array(4, dtype=float32)
                new_item[0] = f_val
                new_item[1] = float32(nx)
                new_item[2] = float32(ny)
                new_item[3] = float32(nz)
                
                device_heap_push(heap[tid], heap_size[tid:], new_item)

def expand_neighbors_fused(
    heap_manager, 
    grid, d_cost, d_pred_action,
    d_distance_field, d_weights, mid_velocity, d_velocity_grid,
    width, height, depth, d_finished_markers, goal):

    num_threads = d_weights.shape[0]
    threads_per_block = 128
    blocks_per_grid = (num_threads + (threads_per_block - 1)) // threads_per_block

    kernel_expand_all_neighbors[blocks_per_grid, threads_per_block](
        heap_manager.heap, heap_manager.heap_size,
        grid, d_cost, d_pred_action,
        d_distance_field, d_weights, mid_velocity, d_velocity_grid,
        width, height, depth, 
        d_finished_markers, goal
    )
    cuda.synchronize()