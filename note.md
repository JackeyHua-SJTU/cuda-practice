`__shared__` can only be inside `__global__` or `__device__` function. It is shared within a thread block.

`__device__` and `__const__` can only be outside kernel functions and in the global region of device files. To const variable, it is accelerated by two ways.
- It can be broadcasted to half-warp if the exact same address is visited
- It can be cached on GPU const cache.

`__global__` is used for kernel function entry. It is run on GPU.

`__host__` can only be run on CPU, and it is the default level of any functions/variables. It is usually used together with `__host__ __device__`.

Never call `__syncthreads()` in a branch, because it is synced via counter. So it will be blocked if it is put inside a `if` branch.

`cudaEventRecord`, think of it as a instruction that record time. It is inserted into instruction queue, and only when GPU has finished all instructions before this one will the timestamp be recorded.

# TODO
- `__shfl_down_sync`
- bank conflict