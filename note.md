## Keyword
`__shared__` can only be inside `__global__` or `__device__` function. It is shared within a thread block.

`__device__` and `__const__` can only be outside kernel functions and in the global region of device files. To const variable, it is accelerated by two ways.
- It can be broadcasted to half-warp if the exact same address is visited
- It can be cached on GPU const cache.

`__global__` is used for kernel function entry. It is run on GPU.

`__host__` can only be run on CPU, and it is the default level of any functions/variables. It is usually used together with `__host__ __device__`.

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

## Quick note

Never call `__syncthreads()` in a branch, because it is synced via counter. So it will be blocked if it is put inside a `if` branch.

`cudaEventRecord`, think of it as a instruction that record time. It is inserted into instruction queue, and only when GPU has finished all instructions before this one will the timestamp be recorded.

# TODO
- `__shfl_down_sync`
- bank conflict