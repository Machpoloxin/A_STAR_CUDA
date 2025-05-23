import numpy as np
from numba import njit, prange, cuda, float32, int32
import math
import time
import numba
from aStarCuda.globalEnv import MAX_NEI


#__device__
@cuda.jit(device=True)
def device_heap_push(heap, heap_size, elem):
    if elem[1] < 0 or elem[2] < 0 or elem[3] < 0:
        return heap_size
    heap[heap_size, 0] = elem[0]
    heap[heap_size, 1] = elem[1]
    heap[heap_size, 2] = elem[2]
    heap[heap_size, 3] = elem[3]
    i = heap_size
    heap_size += 1
    while i > 0:
        parent = (i - 1) // 2
        if heap[parent, 0] > heap[i, 0]:
            temp0 = heap[parent, 0]
            temp1 = heap[parent, 1]
            temp2 = heap[parent, 2]
            temp3 = heap[parent, 3]
            heap[parent, 0] = heap[i, 0]
            heap[parent, 1] = heap[i, 1]
            heap[parent, 2] = heap[i, 2]
            heap[parent, 3] = heap[i, 3]
            heap[i, 0] = temp0
            heap[i, 1] = temp1
            heap[i, 2] = temp2
            heap[i, 3] = temp3
            i = parent
        else:
            break
    return heap_size

@cuda.jit(device=True)
def device_heap_pop(heap, heap_size, output):
    output[0] = heap[0, 0]
    output[1] = heap[0, 1]
    output[2] = heap[0, 2]
    output[3] = heap[0, 3]
    heap_size -= 1
    heap[0, 0] = heap[heap_size, 0]
    heap[0, 1] = heap[heap_size, 1]
    heap[0, 2] = heap[heap_size, 2]
    heap[0, 3] = heap[heap_size, 3]
    i = 0
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < heap_size and heap[left, 0] < heap[smallest, 0]:
            smallest = left
        if right < heap_size and heap[right, 0] < heap[smallest, 0]:
            smallest = right
        if smallest != i:
            temp0 = heap[i, 0]
            temp1 = heap[i, 1]
            temp2 = heap[i, 2]
            temp3 = heap[i, 3]
            heap[i, 0] = heap[smallest, 0]
            heap[i, 1] = heap[smallest, 1]
            heap[i, 2] = heap[smallest, 2]
            heap[i, 3] = heap[smallest, 3]
            heap[smallest, 0] = temp0
            heap[smallest, 1] = temp1
            heap[smallest, 2] = temp2
            heap[smallest, 3] = temp3
            i = smallest
        else:
            break
    return heap_size

#__global__
@cuda.jit
def kernal_heap_push(heap, heapsize, elems):
    tid = cuda.grid(1) # tid = blockIdx.x * blockDim.x + threadIdx.x
    if tid >= heap.shape[0]:
        return
    elem = elems[tid]
    new_size = device_heap_push(heap[tid], heapsize[tid], elem)
    heapsize[tid] = new_size

@cuda.jit
def kernal_heap_pop(heap, heapsize, outputs):
    tid = cuda.grid(1) 
    if tid >= heap.shape[0]:
        return
    new_size = device_heap_pop(heap[tid], heapsize[tid], outputs[tid])
    heapsize[tid] = new_size
    
@cuda.jit
def kernal_heap_delete_last(heap, heapsize, pruning_threshold):
    tid = cuda.grid(1) 
    if tid >= heap.shape[0]:
        return
    if pruning_threshold[tid] > MAX_NEI:
        heapsize[tid] = heapsize[tid] + MAX_NEI - pruning_threshold[tid]
        


class MyHeapManager:
    def __init__(self, num_threads, capacity, init_dtype=np.float32):
        self.heap = cuda.device_array((num_threads, capacity, 4), dtype = init_dtype)
        zeros = np.zeros(num_threads, dtype=np.int32)
        self.heap_size = cuda.to_device(zeros)
    
    def push(self, elems):
        # Push the element to the heap
        threads_per_block = 128
        blocks_per_grid = (self.heap.shape[0] + (threads_per_block - 1)) // threads_per_block # round up
        kernal_heap_push[blocks_per_grid, threads_per_block](self.heap, self.heap_size, elems)
        cuda.synchronize()

    def pop(self):
        # Pop the element from the heap
        # return element with size(#thread,4), and 4: [v, x, y, z], v is the priority value( v = h(x,y,z) )
        threads_per_block = 128
        blocks_per_grid = (self.heap.shape[0] + (threads_per_block - 1)) // threads_per_block
        outputs = cuda.device_array((self.heap.shape[0], 4), dtype = self.heap.dtype)
        kernal_heap_pop[blocks_per_grid, threads_per_block](self.heap, self.heap_size, outputs)
        cuda.synchronize()
        return outputs.copy_to_host()
    
    def delete_last(self, pruning_threshold):
        # pruning the trees base on pruning_threshold[tid] - MAX_NEI
        threads_per_block = 128
        blocks_per_grid = (self.heap.shape[0] + (threads_per_block - 1)) // threads_per_block
        kernal_heap_delete_last[blocks_per_grid, threads_per_block](self.heap, self.heap_size, pruning_threshold)
        cuda.synchronize()
        
    def get_heap_size(self):
        return self.heap_size.copy_to_host()


