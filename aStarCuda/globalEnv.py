import numpy as np

MAX_NEI = 20
INF = 1e9
LOCAL_PATH_LENGTH = 10000

DEFAULT_OPEN_CAPACITY = 100000 

MOVE_OFFSETS = np.array([
    [dx, dy, dz]
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
], dtype=np.int32)