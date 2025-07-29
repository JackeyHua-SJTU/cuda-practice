[Official CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

## Keyword
`__shared__` can only be inside `__global__` or `__device__` function. It is shared within a thread block.

`__device__` and `__const__` can only be outside kernel functions and in the global region of device files. To const variable, it is accelerated by two ways.
- It can be broadcasted to half-warp if the exact same address is visited
- It can be cached on GPU const cache.

`__global__` is used for kernel function entry. It is run on GPU.

`__host__` can only be run on CPU, and it is the default level of any functions/variables. It is usually used together with `__host__ __device__`.


## Memory Layout
This section introduces the layout of GPU memory hierachy.
- Register. Fastest and small. It is automatically used by variable defined in the kernel function. If the register has been used up, then it will use the local memory. It is owned by a thread.
- Local memory. Just a part of global memory. It is used to store the data that exceeds the ability of registers. Slow but private to a thread.
- Shared memory. Just a part of memory on SM. It is private to a thread block. Use `__shared__` to manual allocate on shared memory.
    - **Bank Conflict**. Think of the shared memory as a one-dimentional array, and all `index % 32` will be mapped to the same bank. If there are multiple visit to the same bank, then we need to do it one by one. Latency! 
    - If all warps visit the exact same address, then will broadcast.
- Global memory. Slow but large. Can be accessed by all threads on GPU. Only two explicit ways to allocate on global memory.
    - On the host side, use `cudaMalloc` to allocate a dynamic global variable. Need manually free the resource by `cudaFree`.
    - On the kernel side, use `__device__` to define a global variable. Since it is global, then we can not have multiple same-name global variable. And in other files, we need to use `extern` to declare it.
    ```C++
    __device__ int var;

    int main() {
        int a = 1;
        cudaMemcpySymbol(var, &a, sizeof(a), 0, cudaMemcpyHostToDevice);
    }
    ```

- Constant memory. Pretty like global memory, it is defined in global region of a kernel file. Can be accessed by all threads on GPU. It is slow and not that large. Every SM has a constant cache, which is really fast.

- Every SM has its own L1 and constant cache. SMs share the same L2 cache.


## CUDA Stream
> Key concept in accelerating performance.

Stream is a sequence of operations, and these operations are strictly executed sequentially.
However there is no stream from the side of GPU hardwares. The operations are handled by **COPY engine and KERNEL engine**.
Modern GPU has **two copy engines**, one for device to host, and one for host to device. 
Unless there are dependencies between operations, all engines are executes in **parallel**.
But inside one engine, it is executed sequentially.

So from the side of hardwares, the CUDA driver will first dispatch the operations to each engine, following the launched order.
And it will handle dependency. If there is no dependency, then parallel execution. Otherwise, we may manually wait for prereq to finish.

## Warp Divergence
When branch is encountered, threads INSIDE a warp may go into different clause, e.g. if-else, **if**. Since all threads inside a warp must execute the same code, so stall will be inserted when executing the clause that does not correspond to the thread context. However, if all threads inside a warp fall into the same clause, no stall will be inserted.

> How to optimize?

Aggregate active threads. See [example](./block_reduce.cu).
- In slower case, only partial threads of a warp are active, 
which means the active threads are distributed to a wider range of warps.
- In faster case, active threads are assembled in a few warps, so these warps are more efficient and other _idle_ warps just wait.

> How thread is mapped to a warp?

We have block dim `(x, y, z)`, kinda like `array[z][y][x]`, and we can map the 3 dimention array into a one dimention array.
And we group every consecutive 32 threads into a warp from the very beginning.

> How warp is executed on a SM?

Some warps can be executed in parallel based on the number of warp scheduler. In other cases, warps are executed concurrently. They are dispatched based on clock cycle.


## Performance
Loop Unrolling. See [example](./block_reduce.cu).
- Because every for loop needs a condition judgement each iteration. We spend too much time on useless condition judgement. One way to remedy is to unroll the loop when the round is small and fixed, and do more work in one iteration. The keyword is `#pragma unroll`.

## Warp Shuffle
Thread(register) level data fetching.
- `__shfl_down_sync(mask, var, delta, width=32)`, if it is called at lane id `x`, then it will return the value of variable `var` at lane id `x + delta`. The last `delta` lane will hold its previous value.
- `__shfl_up_sync(mask, var, delta, width=32)`, if it is called at lane id `x`, then it will return the value of variable `var` at lane id `x - delta`. The first `delta` lane will hold its previous value.
- `__shfl_xor_sync(mask, var, laneMask, width=32)`, if it is called at lane id `x`, then it will return the value of variable `var` at lane id `x ^ laneMask`. Usually it is used to swap consecutive number (laneMask = 1), and reduction operation.
- `__shfl_sync(mask, var, srcLane, width=32)` returns the value of variable `var` at lane id `srcLane` to every active lane specified by `mask`.
## Quick note

Never call `__syncthreads()` in a branch, because it is synced via counter. So it will be blocked if it is put inside a `if` branch.

`cudaEventRecord`, think of it as a instruction that record time. It is inserted into instruction queue, and only when GPU has finished all instructions before this one will the timestamp be recorded.

Warp and Engine. As we know, GPU has a kernel engine, and it is comprised of SMs. On each SM, there are several warps that are executing or waiting. Computing resources are on each SM. Kernel engine is a concept from macro view point. Warp is a much micro level concept.

> How threads map to grid and block dim?

Think of the threads as a 1d array `arr`. Denote `n` to be # of threads inside a block. Then we first divide the `arr` every `n` threads. Reshape every segment into grid dim. Inside a segment, reshape the threads into block dim.

**X/Y axis of threadIdx is different from row/column of array.**