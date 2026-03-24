# CUDA Programming Concepts — Beyond the Basics

**Priority:** Week 2. Read before writing tiled matmul.
**Interview map:** Track B Round 4

---

## 1. Memory Coalescing

### What it is
When threads in a warp access **consecutive** memory addresses, the GPU hardware combines (coalesces) these into a single memory transaction. Uncoalesced accesses result in multiple transactions → wasted bandwidth.

### Example
```
32 threads in a warp, each needs a float (4 bytes)

Coalesced (good):
Thread 0 reads addr 0, Thread 1 reads addr 4, Thread 2 reads addr 8, ...
→ One 128-byte transaction. Full bandwidth utilization.

Uncoalesced (bad):
Thread 0 reads addr 0, Thread 1 reads addr 1024, Thread 2 reads addr 2048, ...
→ 32 separate transactions. 32x slower.
```

### Why it matters for matmul
In naive matmul `C[row][col] = sum(A[row][k] * B[k][col])`:
- Reading A row-wise: threads in a warp access consecutive elements → **coalesced**
- Reading B column-wise: threads access elements stride-N apart → **uncoalesced**
- This is one reason naive matmul is slow

### How tiling fixes it
Tiled matmul loads tiles from A and B into shared memory using coalesced reads, then computes from shared memory (which doesn't have coalescing requirements).

---

## 2. Shared Memory Bank Conflicts

### What it is
Shared memory is divided into 32 **banks**. Each bank can serve one address per cycle. If two threads in a warp access the same bank (but different addresses), they serialize → **bank conflict**.

### How banks are organized
```
Address 0  → Bank 0
Address 4  → Bank 1
Address 8  → Bank 2
...
Address 124 → Bank 31
Address 128 → Bank 0  (wraps around)
```

Consecutive 4-byte addresses go to consecutive banks.

### When conflicts happen
```
No conflict: Thread i reads address i*4 → each thread hits a different bank
2-way conflict: Thread i reads address i*8 → threads 0,16 both hit bank 0
                (two threads per bank, 2x slower)
32-way conflict: All threads read same bank → serialized, 32x slower
```

### Exception: broadcast
If multiple threads read the **same address** (not same bank, same address), it's a broadcast — no conflict.

### Why it matters for tiled matmul
When reading from shared memory tiles, the access pattern determines if you get bank conflicts. A common pitfall: if your tile's column stride aligns with 32 banks, every column access creates conflicts. Fix: pad the shared memory array by 1 (`float tile[TILE_SIZE][TILE_SIZE + 1]`).

---

## 3. Occupancy

### What it is
The ratio of active warps to the maximum warps an SM can support.

```
Occupancy = active_warps / max_warps_per_SM
```

An A100 SM supports up to 64 active warps (2048 threads). If your kernel only uses 32 warps per SM, occupancy = 50%.

### Why it matters
Higher occupancy = more warps available for the scheduler to switch between. When one warp stalls (waiting for memory), the scheduler switches to another warp. More warps = better latency hiding.

### What limits occupancy
1. **Registers per thread:** Each SM has a fixed register file (~65K registers). If your kernel uses 64 registers/thread → 65536/64 = 1024 threads max = 32 warps → 50% occupancy.
2. **Shared memory per block:** If your block uses 48 KB of shared memory and the SM has 164 KB, you can fit 3 blocks → limited by block count.
3. **Block size:** If your block has 512 threads and SM max is 2048 → max 4 blocks. If block has 1024 → max 2 blocks.

### The nuance
Higher occupancy doesn't always mean better performance. Sometimes using more registers or shared memory per thread (lower occupancy) gives better instruction-level parallelism and fewer memory accesses. This is called the "occupancy fallacy."

**Rule of thumb:** Aim for ≥50% occupancy, but profile to verify. Don't sacrifice register usage just for occupancy.

---

## 4. Kernel Launch Configuration

### Choosing block size
- Must be a multiple of 32 (warp size)
- Common choices: 128, 256, 512
- 256 is a good default — provides enough threads for parallelism without consuming too many resources
- Too small (32): poor occupancy, wasted SM capacity
- Too large (1024): limits blocks per SM, may limit occupancy

### Choosing grid size
```
gridDim = ceil(problem_size / blockDim)
```
- Must cover all elements
- More blocks = more SMs utilized
- Too few blocks → some SMs idle

### 2D and 3D grids
For matmul, use 2D blocks and grid:
```
dim3 blockDim(TILE_SIZE, TILE_SIZE);  // e.g., 16x16 = 256 threads
dim3 gridDim(N/TILE_SIZE, M/TILE_SIZE);
```

---

## 5. Synchronization

### __syncthreads()
Barrier for all threads in a block. All threads must reach this point before any can proceed.

**When to use:** After cooperative loading into shared memory, before reading from it.

```cuda
// Load tile into shared memory
tile[threadIdx.y][threadIdx.x] = A[row][col];
__syncthreads();  // All threads done loading
// Now safe to read from tile
```

**Pitfall:** If `__syncthreads()` is inside a conditional and not all threads reach it → **deadlock**.

### Atomic operations
`atomicAdd(&addr, val)` — thread-safe addition to global/shared memory.
- Useful for reductions, histograms
- Slow (serializes conflicting threads) — avoid in hot paths
- Better alternatives: warp shuffle, cooperative groups

### Warp-level primitives
- `__shfl_sync()` — exchange data between threads in a warp without shared memory
- `__ballot_sync()` — each thread votes, returns bitmask
- `__reduce_add_sync()` — warp-level reduction
- These are fast because they use the register file, not shared memory

---

## 6. Streams and Async Operations

### CUDA Streams
A stream is a sequence of operations that execute in order on the GPU. Different streams can execute concurrently.

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// These can overlap:
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream2>>>(d_b, d_c);  // Different data, can run concurrently
```

### Why it matters
- Overlap computation with memory transfers
- Keep the GPU busy while waiting for data
- LLM inference engines use streams to overlap prefill and decode operations

### Pinned memory
Regular `malloc` memory must be copied to a staging buffer before GPU transfer. **Pinned (page-locked) memory** skips this step → faster transfers.

```cuda
float *h_a;
cudaMallocHost(&h_a, size);  // Pinned memory
// cudaMemcpyAsync now truly asynchronous
```

---

## 7. Error Handling Pattern

Every CUDA call can fail silently. Always check errors.

```cuda
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Usage:
CUDA_CHECK(cudaMalloc(&d_a, size));
CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

// After kernel launch (kernels don't return error directly):
myKernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

---

## 8. Nsight Compute — What to Look For

### Key metrics when profiling a kernel:

**Memory throughput:**
- `dram__throughput` — HBM bandwidth utilization
- If this is near peak (~2 TB/s on A100) → memory-bound

**Compute throughput:**
- `sm__throughput` — SM compute utilization
- If this is near peak → compute-bound

**Occupancy:**
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- How well you're utilizing SM warp slots

**Memory efficiency:**
- `l2_cache_hit_rate` — higher = less HBM traffic
- `shared_memory_throughput` — are you using shared memory well?

**Roofline analysis:**
- Nsight Compute has a built-in roofline chart
- Shows where your kernel falls: memory-bound or compute-bound
- Tells you which ceiling you're hitting

### How to profile
```bash
ncu --set full -o profile_output ./my_kernel
# Then open profile_output.ncu-rep in Nsight Compute GUI
# Or view in terminal:
ncu --metrics dram__throughput,sm__throughput ./my_kernel
```

---

## Self-Test Questions

1. What is memory coalescing? Why does column-wise access of a matrix hurt performance?
2. What is a bank conflict? How do you fix it in tiled matmul?
3. Define occupancy. What three things limit it?
4. Why doesn't maximum occupancy always mean best performance?
5. When must you use `__syncthreads()`? What happens if it's inside a conditional?
6. What is pinned memory and why does it matter for async transfers?
7. You profile a kernel and see 95% DRAM throughput utilization and 30% SM throughput. Is it memory-bound or compute-bound?

---

## Notes
```


```
