# GPU Architecture — The Foundation

**Priority:** Read this FIRST. Everything else builds on it.
**Interview map:** Track B Round 4 (GPU/Kernel Deep Dive), Track A Round 2 (Technical Deep Dive)

---

## 1. CPU vs GPU — Why GPUs Exist

A CPU has 8–64 powerful cores optimized for sequential tasks — branch prediction, speculative execution, large caches. It's a Swiss Army knife.

A GPU has thousands of simple cores optimized for one thing: doing the same operation on massive amounts of data simultaneously. It trades single-thread performance for throughput.

**Analogy:** CPU = one brilliant mathematician solving problems one at a time. GPU = 10,000 students each solving one simple addition simultaneously.

**Why this matters for LLMs:** Transformer inference involves massive matrix multiplications. A single forward pass through Llama-3 70B requires trillions of multiply-add operations. CPUs can't keep up. GPUs can, because matmul is embarrassingly parallel.

---

## 2. GPU Memory Hierarchy

This is the single most important concept for understanding GPU performance. Memorize these numbers.

```
┌─────────────────────────────────────────────────────────────┐
│                    Registers                                 │
│                    Per thread                                │
│                    ~20 TB/s effective bandwidth               │
│                    ~256 KB per SM                             │
│                    Latency: ~1 cycle                         │
├─────────────────────────────────────────────────────────────┤
│                    Shared Memory (SRAM)                      │
│                    Per block — shared among threads           │
│                    ~19 TB/s on H100                           │
│                    ~228 KB per SM on A100                     │
│                    Latency: ~20-30 cycles                    │
├─────────────────────────────────────────────────────────────┤
│                    L2 Cache                                   │
│                    Shared across all SMs                      │
│                    ~6 MB on A100, ~50 MB on H100              │
│                    ~5 TB/s                                    │
│                    Latency: ~200 cycles                      │
├─────────────────────────────────────────────────────────────┤
│                    HBM (Global Memory)                        │
│                    The big, slow memory                       │
│                    80 GB on A100, 80 GB on H100               │
│                    ~2 TB/s on A100, ~3.35 TB/s on H100        │
│                    Latency: ~400-600 cycles                   │
└─────────────────────────────────────────────────────────────┘
```

### Key insight: There's a ~10,000x difference between register and HBM access speed.

Every optimization in GPU computing — tiling, kernel fusion, FlashAttention — exists to keep data in faster memory and minimize trips to HBM.

---

## 3. Streaming Multiprocessors (SMs)

A GPU is organized into **Streaming Multiprocessors (SMs)**.

- A100 has **108 SMs**
- H100 has **132 SMs**
- Each SM has its own:
  - Registers
  - Shared memory (SRAM)
  - Warp schedulers
  - CUDA cores (FP32/FP64)
  - Tensor cores (matrix multiply hardware)

When you launch a kernel, CUDA assigns **blocks to SMs**. Each SM can run multiple blocks concurrently. Threads within a block can share data via shared memory.

---

## 4. Warps — The Actual Unit of Execution

A **warp** = 32 threads that execute the **same instruction at the same time** (SIMT — Single Instruction Multiple Threads).

This is critical:
- If all 32 threads in a warp take the same branch → full speed
- If some threads take `if` and others take `else` → **warp divergence** → both branches execute sequentially → performance halved (or worse)

### Why this matters for LLMs
Causal attention masking creates a triangular pattern — some threads compute attention, others are masked out. This causes warp divergence. FlashAttention handles this efficiently by restructuring the computation.

### Interview question this answers:
> "What is warp divergence and how does it affect attention computation with causal masking?"

---

## 5. Tensor Cores

Tensor Cores are specialized hardware units that perform small matrix multiplications (e.g., 4x4 or 16x16) in a single operation.

- A100: 3rd gen Tensor Cores — FP16, BF16, TF32, INT8, FP64
- H100: 4th gen Tensor Cores — adds FP8, doubles throughput

**Why they matter:**
- Regular CUDA cores: one multiply-add per core per cycle
- Tensor cores: a 4x4 matrix multiply-add in one operation = 64 multiply-adds per cycle

**Practical implication:** If your code doesn't use Tensor Cores (e.g., you wrote a naive matmul), you're leaving >10x performance on the table. TRT-LLM and FlashAttention are designed to maximize Tensor Core utilization.

---

## 6. Compute-Bound vs Memory-Bound

Every GPU kernel is limited by one of two things:
1. **Compute-bound:** GPU math units are the bottleneck. You're doing so many operations that the GPU can't compute fast enough.
2. **Memory-bound:** GPU memory bandwidth is the bottleneck. You're waiting for data to arrive from HBM.

### Arithmetic Intensity
```
Arithmetic Intensity = FLOPs / Bytes Moved
```

- **High arithmetic intensity** (many FLOPs per byte) → compute-bound
- **Low arithmetic intensity** (few FLOPs per byte) → memory-bound

### The Roofline Model

```
Performance
(FLOPS)
    │
    │         ╱ Compute ceiling (peak FLOPS)
    │        ╱─────────────────────────
    │       ╱
    │      ╱
    │     ╱  ← Memory bandwidth ceiling (slope)
    │    ╱
    │   ╱
    │  ╱
    │ ╱
    └──────────────────────────────────
         Arithmetic Intensity (FLOPs/Byte)
```

**Where LLM operations fall:**

| Operation | Arithmetic Intensity | Bound |
|---|---|---|
| Matrix-matrix multiply (prefill) | High | Compute-bound |
| Matrix-vector multiply (decode) | Low | Memory-bound |
| Attention (long sequences) | Medium-High | Depends on implementation |
| Softmax | Low | Memory-bound |
| LayerNorm | Low | Memory-bound |

### Interview question this answers:
> "What is the arithmetic intensity of a matrix-vector multiply vs matrix-matrix multiply? Which regime is LLM decode in?"

**Answer:** Matrix-vector multiply has O(n²) operations but loads O(n²) data → arithmetic intensity ≈ 1. Matrix-matrix multiply has O(n³) operations but loads O(n²) data → arithmetic intensity ≈ n. LLM decode (generating one token at a time) is a matrix-vector multiply against the model weights → memory-bound.

---

## 7. Why LLM Decode is Memory-Bandwidth-Bound

This is THE question they will ask. Understand it deeply.

### During prefill (processing the prompt):
- Input is a matrix (batch_size × seq_len × hidden_dim)
- Each layer does matrix-matrix multiplications
- Arithmetic intensity is high → **compute-bound**
- GPU Tensor Cores are the bottleneck

### During decode (generating tokens one at a time):
- Input is a vector (batch_size × 1 × hidden_dim) — ONE new token
- Each layer does matrix-vector multiplications
- Must load ALL model weights from HBM for every single token
- Llama-3 70B = ~140 GB of weights in FP16
- A100 HBM bandwidth = 2 TB/s
- Minimum time to load weights = 140 GB / 2 TB/s = 70 ms per token
- But the actual computation (multiply-add) takes << 70 ms
- → **Memory-bandwidth-bound**

### Why batching helps throughput but not latency:
- With batch size 1: load 140 GB of weights, do 1 token's worth of computation → wasted bandwidth
- With batch size 32: load 140 GB of weights ONCE, do 32 tokens' worth of computation → 32x better utilization of loaded data
- Each individual token still waits the same time → TTFT doesn't improve
- But total tokens/second increases → throughput improves

### Interview question this answers:
> "Why does batching help LLM inference throughput but not necessarily TTFT?"

---

## 8. Key Numbers to Memorize

### A100 80GB
- HBM: 80 GB, 2 TB/s bandwidth
- FP16 Tensor Core: 312 TFLOPS
- FP32: 19.5 TFLOPS
- SMs: 108
- Shared memory: 164 KB per SM (configurable, up to 228 KB)
- L2 cache: 40 MB

### H100 80GB
- HBM: 80 GB, 3.35 TB/s bandwidth
- FP16 Tensor Core: 989 TFLOPS
- FP8 Tensor Core: 1,979 TFLOPS
- SMs: 132
- Shared memory: 228 KB per SM
- L2 cache: 50 MB

### Rough Estimates for LLM Serving
- Llama-3 8B in FP16: ~16 GB VRAM → fits on single A100
- Llama-3 70B in FP16: ~140 GB VRAM → needs 2x A100 80GB (tensor parallelism)
- Llama-3 70B in INT8: ~70 GB VRAM → fits on single A100 80GB
- Llama-3 70B in FP8: ~70 GB VRAM → fits on single H100 80GB

---

## Self-Test Questions

Answer these without looking back. If you can't, re-read that section.

1. Draw the GPU memory hierarchy from fastest to slowest. Give approximate bandwidth for each level on A100.
2. What is a warp? How many threads? What happens when threads in a warp diverge?
3. What is arithmetic intensity? Give the formula.
4. Why is LLM decode memory-bound but prefill is compute-bound?
5. A customer has an A100 80GB and wants to serve Llama-3 70B. Can it fit? What are their options?
6. Why does increasing batch size help throughput during decode?
7. What are Tensor Cores and why are they important?
8. What is the roofline model? Sketch one and place "prefill matmul" and "decode matmul" on it.

---

## Notes
```
(Take your own notes here)


```
