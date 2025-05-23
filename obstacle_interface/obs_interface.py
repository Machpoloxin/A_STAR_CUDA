
import numpy as np
from numba import njit, prange, cuda, float32, int32
import math
import time
import numba

@cuda.jit(device=True)
def distance_to_oriented_cuboid_device(point, center, dims, quat):
    '''
    Compute the distance from a point to an oriented cuboid.
    :param point: 3D point (x, y, z) !!! !!! 
    :param center: 3D center of the cuboid (x, y, z)
    :param dims: 3D dimensions of the cuboid (length, width, height)
    :param quat: 4D quaternion (x, y, z, w)
    :return: Euclidean distance to the cuboid
    '''
    # Compute rotation matrix components from quaternion (x, y, z, w)
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]
    r00 = 1 - 2*y*y - 2*z*z
    r01 = 2*x*y - 2*z*w
    r02 = 2*x*z + 2*y*w

    r10 = 2*x*y + 2*z*w
    r11 = 1 - 2*x*x - 2*z*z
    r12 = 2*y*z - 2*x*w

    r20 = 2*x*z - 2*y*w
    r21 = 2*y*z + 2*x*w
    r22 = 1 - 2*x*x - 2*y*y

    # Translate point to cuboid's local frame:
    rel0 = point[0] - center[0]
    rel1 = point[1] - center[1]
    rel2 = point[2] - center[2]
    local0 = r00 * rel0 + r10 * rel1 + r20 * rel2
    local1 = r01 * rel0 + r11 * rel1 + r21 * rel2
    local2 = r02 * rel0 + r12 * rel1 + r22 * rel2

    half0 = dims[0] * 0.5
    half1 = dims[1] * 0.5
    half2 = dims[2] * 0.5

    dx = 0.0
    dy = 0.0
    dz = 0.0
    if math.fabs(local0) > half0:
        dx = math.fabs(local0) - half0
    if math.fabs(local1) > half1:
        dy = math.fabs(local1) - half1
    if math.fabs(local2) > half2:
        dz = math.fabs(local2) - half2
    return math.sqrt(dx*dx + dy*dy + dz*dz)


@cuda.jit
def compute_distance_field_kernel(distance_field, depth, height, width,
                                  cube_centers, cube_dims, cube_rotations,
                                  cuboid_centers, cuboid_dims, cuboid_rotations):
    '''
    Compute the distance field for a 3D grid using the GPU.
    :param distance_field: 3D array to store the distance field [z,y,x]!!!
    :param depth: Depth of the 3D grid
    :param height: Height of the 3D grid
    :param width: Width of the 3D grid
    :param cube_centers: Array of cube centers
    :param cube_dims: Array of cube dimensions
    :param cube_rotations: Array of cube rotations
    :param cuboid_centers: Array of cuboid centers
    :param cuboid_dims: Array of cuboid dimensions
    :param cuboid_rotations: Array of cuboid rotations
    '''
    x, y, z = cuda.grid(3)
    if x < width and y < height and z < depth:
        point = cuda.local.array(3, dtype=numba.float32)
        point[0] = float(x)
        point[1] = float(y)
        point[2] = float(z)
        
        min_dist = 1e6
        n_cubes = cube_centers.shape[0]
        for i in range(n_cubes):
            d = distance_to_oriented_cuboid_device(point, cube_centers[i], cube_dims[i], cube_rotations[i])
            if d < min_dist:
                min_dist = d
        n_cuboids = cuboid_centers.shape[0]
        for i in range(n_cuboids):
            d = distance_to_oriented_cuboid_device(point, cuboid_centers[i], cuboid_dims[i], cuboid_rotations[i])
            if d < min_dist:
                min_dist = d
        distance_field[z, y, x] = min_dist
    

def precompute_distance_field(depth, height, width,
                              cube_centers, cube_dims, cube_rotations,
                              cuboid_centers, cuboid_dims, cuboid_rotations):
    '''
    Host function to precompute the distance field in a 3D grid using the GPU.
    :param depth: Depth of the 3D grid
    :param height: Height of the 3D grid
    :param width: Width of the 3D grid
    :param cube_centers: Array of cube centers
    :param cube_dims: Array of cube dimensions
    :param cube_rotations: Array of cube rotations
    :param cuboid_centers: Array of cuboid centers
    :param cuboid_dims: Array of cuboid dimensions
    :param cuboid_rotations: Array of cuboid rotations
    :return: 3D distance field [z,y,x] stored in a numpy array
    '''
    distance_field = np.empty((depth, height, width), dtype=np.float64)
    threads_per_block = (8, 8, 8)
    blocks_per_grid_x = (depth + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_z = (width + threads_per_block[2] - 1) // threads_per_block[2]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    
    d_cube_centers = cuda.to_device(cube_centers)
    d_cube_dims = cuda.to_device(cube_dims)
    d_cube_rotations = cuda.to_device(cube_rotations)
    d_cuboid_centers = cuda.to_device(cuboid_centers)
    d_cuboid_dims = cuda.to_device(cuboid_dims)
    d_cuboid_rotations = cuda.to_device(cuboid_rotations)
    d_distance_field = cuda.to_device(distance_field)
    compute_distance_field_kernel[blocks_per_grid, threads_per_block](d_distance_field, depth, height, width,
                                                                       d_cube_centers, d_cube_dims, d_cube_rotations,
                                                                       d_cuboid_centers, d_cuboid_dims, d_cuboid_rotations)
    d_distance_field.copy_to_host(distance_field)
    return distance_field

def pack_obstacle_data(obstacles, sq_side_length):
    cube_centers = []
    cube_dims = []
    cube_rotations = []
    cuboid_centers = []
    cuboid_dims = []
    cuboid_rotations = []
    for obs in obstacles:
        if obs["type"] == "hand":
            center = np.array(obs["hand_position"], dtype=np.float64)
            radius = obs["length"]
            dims = np.array([radius, radius, radius], dtype=np.float64)
            quat = np.array(obs["orientation"], dtype=np.float64)
            cube_centers.append(center)
            cube_dims.append(dims)
            cube_rotations.append(quat)
        elif obs["type"] == "cuboid":
            center = np.array(obs["mid_point"], dtype=np.float64)
            length = obs["length"]
            dims = np.array([length, sq_side_length, sq_side_length], dtype=np.float64)
            quat = np.array(obs["orientation"], dtype=np.float64)
            cuboid_centers.append(center)
            cuboid_dims.append(dims)
            cuboid_rotations.append(quat)
    return (np.array(cube_centers), np.array(cube_dims), np.array(cube_rotations),
            np.array(cuboid_centers), np.array(cuboid_dims), np.array(cuboid_rotations))
