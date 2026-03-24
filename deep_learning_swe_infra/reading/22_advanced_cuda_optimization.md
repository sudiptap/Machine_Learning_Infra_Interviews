# Advanced CUDA Optimization Patterns

**Priority:** Week 6-8.
**Interview map:** Track B Round 4

---

## 1. Kernel Fusion — The Most Important Optimization

### What it is
Combine multiple operations into a single kernel to eliminate intermediate HBM reads/writes.

### Example: Fused Add + LayerNorm
```
Unfused (3 kernels, 3 HBM round trips):
  Kernel 1: residual = x + attention_output      → write to HBM
  Kernel 2: mean, var = compute_stats(residual)   → read from HBM, write to HBM
  Kernel 3: output = (residual - mean) / sqrt(var) * gamma + beta  → read from HBM

Fused (1 kernel, 1 HBM round trip):
  Kernel: load x and attention_output
          compute residual = x + attention_output
          compute mean, var (in registers/shared mem)
          compute normalized output
          write final result to HBM
```

**Impact:** 3x fewer HBM accesses for these operations. For memory-bound operations like LayerNorm, this is nearly a 3x speedup.

### What gets fused in practice
- **Attention:** Q×K^T → scale → mask → softmax → ×V (FlashAttention = fully fused)
- **MLP:** Linear → GeLU/SiLU activation (fused by TRT-LLM)
- **LayerNorm:** Residual add + LayerNorm (fused by most frameworks)
- **Rotary position embedding:** RoPE applied during attention kernel (not separate)

### Why TRT-LLM is fast
TRT-LLM's compile step identifies all fusion opportunities for a specific model and selects pre-optimized fused kernels. This is why compile-time optimization beats runtime.

---

## 2. Memory Access Patterns

### Strided vs Contiguous Access
```
Contiguous (good): thread i reads element i
  Memory: [0][1][2][3][4][5][6][7]
  Thread:  0  1  2  3  4  5  6  7
  → 1 memory transaction

Strided (bad): thread i reads element i*stride
  Memory: [0][ ][ ][ ][4][ ][ ][ ][8]...
  Thread:  0              1              2
  → Multiple transactions, wasted bandwidth
```

### Structure of Arrays (SoA) vs Array of Structures (AoS)
```
AoS (bad for GPU):
  struct Particle { float x, y, z, w; };
  Particle particles[N];
  // Thread i reads particles[i].x — strided by sizeof(Particle)

SoA (good for GPU):
  float x[N], y[N], z[N], w[N];
  // Thread i reads x[i] — contiguous!
```

**For LLM inference:** Weight matrices are stored in formats that ensure coalesced access during matmul. Layout matters enormously.

---

## 3. Warp-Level Primitives

### Warp Shuffle
Exchange data between threads in a warp without shared memory:
```cuda
// Thread i has value v
// Get value from thread (i + 1) % 32:
float v_next = __shfl_down_sync(0xffffffff, v, 1);

// Warp-level reduction (sum):
for (int offset = 16; offset > 0; offset /= 2) {
    v += __shfl_down_sync(0xffffffff, v, offset);
}
// Thread 0 now has the sum of all 32 threads
```

**Why it matters:** Faster than shared memory for small reductions. Used in softmax, LayerNorm, and attention score computation.

### Cooperative Groups (CUDA 9+)
More flexible grouping than warps:
```cuda
// Tile of 16 threads (half warp)
auto tile = cg::tiled_partition<16>(cg::this_thread_block());
tile.sync();  // Only sync these 16 threads
float sum = cg::reduce(tile, val, cg::plus<float>());
```

---

## 4. Asynchronous Memory Operations

### Async Copy (CUDA 11+, Ampere+)
Copy data from global memory to shared memory without going through registers:
```cuda
// Old way (2 steps):
float val = global_mem[i];     // Global → register
shared_mem[i] = val;           // Register → shared

// New way (1 step, async):
__pipeline_memcpy_async(&shared_mem[i], &global_mem[i], sizeof(float));
__pipeline_commit();
// ... do other work ...
__pipeline_wait_prior(0);     // Wait for copy to complete
```

**Why it matters:** Overlaps memory transfer with computation. Used in FlashAttention and tiled matmul implementations for maximum performance.

### TMA (Tensor Memory Accelerator, Hopper/H100)
Hardware unit that handles complex memory access patterns:
- Loads 2D tiles from global memory directly
- Supports multicast (load once, distribute to multiple SMs)
- FlashAttention-3 uses TMA for significant speedup on H100

---

## 5. Quantized Computation Patterns

### INT8 Matrix Multiply
```
Standard: C[float32] = A[float16] × B[float16]
INT8:     C[int32] = A[int8] × B[int8]          → then scale to float

Steps:
1. Quantize A, B to INT8 (apply scale factors)
2. Compute INT8 matmul on Tensor Cores (2x throughput)
3. Dequantize result back to float (apply inverse scale)
```

### FP8 on H100
```
C[float32] = A[fp8_e4m3] × B[fp8_e4m3]

H100 Tensor Cores natively support FP8:
- 2x throughput vs FP16 Tensor Cores
- Less precision loss than INT8 (floating point preserves dynamic range)
- Per-tensor or per-channel scaling
```

### Mixed-Precision Patterns
```
Weights: INT4/INT8/FP8 (stored compressed)
Activations: FP16/BF16 (full precision for computation)
Accumulation: FP32 (prevent overflow during matmul)
Output: FP16/BF16 (for next layer)

Flow: dequantize weights → FP16 matmul → FP32 accumulate → cast to FP16
```

---

## 6. Attention Kernel Optimization

### Multi-Head Attention (MHA) Kernel Strategies

**Batched GEMM approach:**
- Reshape Q, K, V to batch all heads together
- Use cuBLAS batched GEMM for Q×K^T and (softmax)×V
- Simple but not fused — softmax is separate kernel

**Fused MHA (fMHA):**
- NVIDIA's proprietary fused attention kernel
- Combines Q×K^T, scaling, masking, softmax, ×V in one kernel
- Used by TRT-LLM
- Highly optimized for specific head dimensions (64, 128)

**FlashAttention approach:**
- Tiled computation in SRAM
- Online softmax for streaming computation
- Open source, available in CUDA and Triton
- Matches or exceeds fMHA for most configurations

### GQA Kernel Optimization
```
GQA: 32 Q heads, 8 KV heads → 4 Q heads per KV head

Naive: expand K,V to match Q head count, then standard MHA
  → Wastes memory and compute

Optimized: group Q heads, share K,V reads
  → Each K,V loaded once, used for 4 Q heads
  → 4x less K,V memory bandwidth
  → FlashInfer and vLLM implement this efficiently
```

---

## 7. Memory Management Patterns

### Memory Pools
Instead of calling cudaMalloc for every allocation (expensive):
```
// Pre-allocate a large pool at startup
void* pool = cudaMalloc(POOL_SIZE);

// Suballocate from pool (fast, just pointer arithmetic)
void* block1 = pool_alloc(pool, size1);
void* block2 = pool_alloc(pool, size2);

// Free returns to pool (no cudaFree)
pool_free(pool, block1);
```
vLLM's PagedAttention uses this pattern for KV cache blocks.

### Pinned Memory for Swapping
When swapping KV cache to CPU:
```
// Pinned memory enables async CPU↔GPU transfer
cudaMallocHost(&cpu_kv_cache, size);  // Pinned

// Async transfer (doesn't block GPU)
cudaMemcpyAsync(cpu_kv_cache, gpu_kv_cache, size, cudaMemcpyDeviceToHost, stream);
// GPU continues computing while transfer happens
```

---

## 8. Profiling-Driven Optimization

### The optimization loop
```
1. Profile the kernel (Nsight Compute)
2. Identify the bottleneck (memory-bound? compute-bound? latency-bound?)
3. Apply the appropriate optimization:
   - Memory-bound → reduce HBM accesses (fusion, tiling, caching)
   - Compute-bound → use Tensor Cores, reduce FLOPs (quantization)
   - Latency-bound → increase occupancy, use async operations
4. Re-profile and verify improvement
5. Repeat
```

### Key Nsight Compute metrics to check
```
Memory-bound indicators:
  dram__throughput.avg.pct_of_peak > 80%  → you're hitting HBM limit
  sm__throughput.avg.pct_of_peak < 50%    → GPU compute is underused

Compute-bound indicators:
  sm__throughput.avg.pct_of_peak > 80%    → GPU compute is saturated
  dram__throughput.avg.pct_of_peak < 50%  → memory is not the limit

Latency-bound indicators:
  Both throughputs low                     → stalls, low occupancy
  sm__warps_active.avg.pct < 40%          → not enough warps to hide latency
```

---

## Self-Test Questions

1. What is kernel fusion and why does it help memory-bound operations?
2. Give an example of operations that get fused in LLM inference.
3. What is a warp shuffle and when would you use it over shared memory?
4. How does async copy (CUDA 11+) improve performance?
5. Explain the mixed-precision pattern for INT4 weight inference.
6. How does GQA kernel optimization avoid redundant K,V reads?
7. You profile a kernel and find 90% DRAM throughput, 30% SM throughput. What do you do?

---

## Notes
```


```
