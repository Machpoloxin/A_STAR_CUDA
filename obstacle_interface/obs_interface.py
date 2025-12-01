import numpy as np
from numba import cuda, float32
import math

@cuda.jit(device=True)
def distance_to_oriented_cuboid_device(px, py, pz, cx, cy, cz, dx, dy, dz, qx, qy, qz, qw):
    '''
    Compute distance using scalars to avoid array view issues in Numba.
    '''
    # Compute rotation matrix components
    x, y, z, w = qx, qy, qz, qw
    r00 = 1 - 2*y*y - 2*z*z
    r01 = 2*x*y - 2*z*w
    r02 = 2*x*z + 2*y*w

    r10 = 2*x*y + 2*z*w
    r11 = 1 - 2*x*x - 2*z*z
    r12 = 2*y*z - 2*x*w

    r20 = 2*x*z - 2*y*w
    r21 = 2*y*z + 2*x*w
    r22 = 1 - 2*x*x - 2*y*y

    # Translate point
    rel0 = px - cx
    rel1 = py - cy
    rel2 = pz - cz

    # Rotate point
    local0 = r00 * rel0 + r10 * rel1 + r20 * rel2
    local1 = r01 * rel0 + r11 * rel1 + r21 * rel2
    local2 = r02 * rel0 + r12 * rel1 + r22 * rel2

    half0 = dx * 0.5
    half1 = dy * 0.5
    half2 = dz * 0.5

    d_x = 0.0
    d_y = 0.0
    d_z = 0.0

    if math.fabs(local0) > half0:
        d_x = math.fabs(local0) - half0
    if math.fabs(local1) > half1:
        d_y = math.fabs(local1) - half1
    if math.fabs(local2) > half2:
        d_z = math.fabs(local2) - half2

    return math.sqrt(d_x*d_x + d_y*d_y + d_z*d_z)


@cuda.jit
def compute_distance_field_kernel(distance_field, depth, height, width,
                                  cube_centers, cube_dims, cube_rotations,
                                  cuboid_centers, cuboid_dims, cuboid_rotations):
    x, y, z = cuda.grid(3)
    if x >= width or y >= height or z >= depth:
        return

    px = float32(x)
    py = float32(y)
    pz = float32(z)
    
    min_dist = 1e9
    n_cubes = cube_centers.shape[0]
    for i in range(n_cubes):
        cx = cube_centers[i, 0]
        cy = cube_centers[i, 1]
        cz = cube_centers[i, 2]
        
        cdx = cube_dims[i, 0]
        cdy = cube_dims[i, 1]
        cdz = cube_dims[i, 2]
        
        qx = cube_rotations[i, 0]
        qy = cube_rotations[i, 1]
        qz = cube_rotations[i, 2]
        qw = cube_rotations[i, 3]

        d = distance_to_oriented_cuboid_device(px, py, pz, cx, cy, cz, cdx, cdy, cdz, qx, qy, qz, qw)
        if d < min_dist:
            min_dist = d

    # Process Cuboids
    n_cuboids = cuboid_centers.shape[0]
    for i in range(n_cuboids):
        cx = cuboid_centers[i, 0]
        cy = cuboid_centers[i, 1]
        cz = cuboid_centers[i, 2]
        
        cdx = cuboid_dims[i, 0]
        cdy = cuboid_dims[i, 1]
        cdz = cuboid_dims[i, 2]
        
        qx = cuboid_rotations[i, 0]
        qy = cuboid_rotations[i, 1]
        qz = cuboid_rotations[i, 2]
        qw = cuboid_rotations[i, 3]

        d = distance_to_oriented_cuboid_device(px, py, pz, cx, cy, cz, cdx, cdy, cdz, qx, qy, qz, qw)
        if d < min_dist:
            min_dist = d

    distance_field[z, y, x] = min_dist

def precompute_distance_field(depth, height, width,
                              cube_centers, cube_dims, cube_rotations,
                              cuboid_centers, cuboid_dims, cuboid_rotations):
    distance_field = np.empty((depth, height, width), dtype=np.float32)
    
    d_cc = cuda.to_device(np.ascontiguousarray(cube_centers, dtype=np.float32))
    d_cd = cuda.to_device(np.ascontiguousarray(cube_dims, dtype=np.float32))
    d_cr = cuda.to_device(np.ascontiguousarray(cube_rotations, dtype=np.float32))
    
    d_bc = cuda.to_device(np.ascontiguousarray(cuboid_centers, dtype=np.float32))
    d_bd = cuda.to_device(np.ascontiguousarray(cuboid_dims, dtype=np.float32))
    d_br = cuda.to_device(np.ascontiguousarray(cuboid_rotations, dtype=np.float32))
    
    d_df = cuda.to_device(distance_field)

    threads_per_block = (8, 8, 8)
    blocks_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_z = (depth + threads_per_block[2] - 1) // threads_per_block[2]
    blocks_per_grid = (blocks_x, blocks_y, blocks_z)

    compute_distance_field_kernel[blocks_per_grid, threads_per_block](
        d_df, depth, height, width,
        d_cc, d_cd, d_cr,
        d_bc, d_bd, d_br
    )
    
    d_df.copy_to_host(distance_field)
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
            cube_centers.append(obs["hand_position"])
            r = obs["length"]
            cube_dims.append((r, r, r))
            cube_rotations.append(obs["orientation"])
        elif obs["type"] == "cuboid":
            cuboid_centers.append(obs["mid_point"])
            l = obs["length"]
            cuboid_dims.append((l, sq_side_length, sq_side_length))
            cuboid_rotations.append(obs["orientation"])

    def to_np(data, cols):
        if not data:
            return np.zeros((0, cols), dtype=np.float32)
        return np.array(data, dtype=np.float32).reshape(-1, cols)

    return (
        to_np(cube_centers, 3), to_np(cube_dims, 3), to_np(cube_rotations, 4),
        to_np(cuboid_centers, 3), to_np(cuboid_dims, 3), to_np(cuboid_rotations, 4)
    )