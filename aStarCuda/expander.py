
import numpy as np
from numba import njit, cuda, float32, int32
from aStarCuda.heuristic import device_calculate_heuristic_f
from aStarCuda.globalEnv import MAX_NEI, INF

@cuda.jit(device=True)
def device_expand_neighbors(
    cur_coordinates, goal,
    grid, cost, pred_x, pred_y, pred_z, pred_v,
    distance_field, weights, mid_velocity,  vel, dx, dy, dz,
    width, height, depth, new_elem, d_cur_add_elems
):
    cx = cur_coordinates[0]
    cy = cur_coordinates[1]
    cz = cur_coordinates[2]
    if cx == goal[0] and cy == goal[1] and cz == goal[2]:
      return

    if dx==0 and dy==0 and dz==0:
      return
    nx, ny, nz = cx+dx, cy+dy, cz+dz
    # boundary check
    if not (0<=nx<width and 0<=ny<height and 0<=nz<depth):
      return
    if grid[nz,ny,nx] != 0:
      return
    # move cost
    s = abs(dx)+abs(dy)+abs(dz)
    if s == 1:
        move_cost = 1.0
    elif s == 2:
        move_cost = 1.41
    else:
        move_cost = 1.73

    # new cost
    base = cost[cz,cy,cx]
    newc = base + move_cost/(vel+1.0)*weights[0]

    if newc < cost[nz,ny,nx]:
      d_cur_add_elems = d_cur_add_elems + 1
      cost[nz,ny,nx] = newc
      dist_to_obs = distance_field[nz,ny,nx]
      
      newpri = device_calculate_heuristic_f(cost[nz,ny,nx], mid_velocity, vel, 
                                              dist_to_obs, weights, cur_coordinates , goal)
      new_elem[0] = newpri
      new_elem[1] = nx
      new_elem[2] = ny
      new_elem[3] = nz

      pred_x[nz,ny,nx] = cx
      pred_y[nz,ny,nx] = cy
      pred_z[nz,ny,nx] = cz
      pred_v[nz,ny,nx] = vel
    

@cuda.jit
def kernel_expand_neighbors(cur_coordinates, goal,
    grid, cur_cost, pred_x, pred_y, pred_z, pred_v,
    distance_field, weights, mid_velocity, vel, dx, dy, dz,
    width, height, depth, out_elems, d_finished_markers, d_cur_add_elems):

    tid = cuda.grid(1)
    if tid >= cur_coordinates.shape[0] or d_finished_markers[tid] == 1:
      return
    
    if (cur_coordinates[tid, 0] == goal[0] and
        cur_coordinates[tid, 1] == goal[1] and
        cur_coordinates[tid, 2] == goal[2]):
      d_finished_markers[tid] = 1
      return
    
    device_expand_neighbors(cur_coordinates[tid], goal,
                            grid, cur_cost[tid], pred_x[tid], pred_y[tid], pred_z[tid], pred_v[tid],
                            distance_field, weights[tid], mid_velocity, vel, dx, dy, dz,
                            width, height, depth, out_elems[tid], d_cur_add_elems[tid])
            
            
def expand_neighbors(fullBinThree, cur_coordinates, goal,
    grid, cur_cost, pred_x, pred_y, pred_z, pred_v,
    distance_field, weights, mid_velocity, velocity_grid,
    width, height, depth, d_finished_markers, MAX_NEI=MAX_NEI):

    num_threads = cur_coordinates.shape[0]
    threads_per_block = 128
    blocks_per_grid = (num_threads + (threads_per_block - 1)) // threads_per_block # round up
    new_elems = -1 * np.ones((weights.shape[0],4), dtype=np.float32)
    new_elems[:,0] = INF
    cur_add_elems = np.zeros((weights.shape[0],), dtype=np.int32)
    d_cur_add_elems = cuda.to_device(cur_add_elems)
    for dx in (-1,0,1):
      for dy in (-1,0,1):
        for dz in (-1,0,1):
          for vi in range(velocity_grid.shape[0]):
            vel = velocity_grid[vi]
            d_new_elems = cuda.to_device(new_elems)
            kernel_expand_neighbors[blocks_per_grid, threads_per_block](
                cur_coordinates, goal,
                grid, cur_cost, pred_x, pred_y, pred_z, pred_v,
                distance_field, weights, mid_velocity, vel, dx, dy, dz,
                width, height, depth, d_new_elems, d_finished_markers, d_cur_add_elems
            )
            fullBinThree.push(d_new_elems)
    cuda.synchronize()
    fullBinThree.delete_last(d_cur_add_elems)
    cuda.synchronize()

    return fullBinThree


