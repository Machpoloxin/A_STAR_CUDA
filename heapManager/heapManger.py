import numpy as np
from numba import cuda, float32, int32

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
            t0 = heap[parent, 0]; t1 = heap[parent, 1]; t2 = heap[parent, 2]; t3 = heap[parent, 3]
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
def kernel_heap_push(heap, heap_size, elems):
    tid = cuda.grid(1)
    if tid < heap.shape[0]:
        # heap_size[tid:tid+1] 
        device_heap_push(heap[tid], heap_size[tid:tid+1], elems[tid])

class MyHeapManager:
    def __init__(self, num_threads, capacity):
        # Heap structure: (N_threads, Capacity, 4)
        self.heap = cuda.device_array((num_threads, capacity, 4), dtype=np.float32)
        # Heap size counter for each thread
        self.heap_size = cuda.to_device(np.zeros(num_threads, dtype=np.int32))
        self.capacity = capacity

    def push(self, elems):
        num_threads = self.heap.shape[0]
        threads_per_block = 128
        blocks_per_grid = (num_threads + threads_per_block - 1) // threads_per_block
        
        kernel_heap_push[blocks_per_grid, threads_per_block](
            self.heap, 
            self.heap_size, 
            elems
        )
        cuda.synchronize()

    def get_heap_size(self):
        return self.heap_size.copy_to_host()