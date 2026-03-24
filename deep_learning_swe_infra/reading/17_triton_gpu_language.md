# Triton GPU Language — Concepts Before Hands-On

**Priority:** Week 4-5. Read before writing Triton kernels.
**Interview map:** Track B Rounds 3 & 4

---

## 1. What is Triton?

**OpenAI Triton** (not NVIDIA Triton Inference Server — confusingly same name, completely different things).

- A **Python-based GPU programming language** for writing custom kernels
- Compiles Python-like code to optimized GPU machine code (PTX)
- Sits between CUDA (low-level, C++) and PyTorch (high-level, no kernel control)
- Created by Philippe Tillet at OpenAI

```
Abstraction levels:

PyTorch:   model.forward(x)        ← No kernel control, framework decides
Triton:    @triton.jit def kernel() ← Block-level control, Python syntax
CUDA:      __global__ void kernel() ← Thread-level control, C++ syntax
PTX/SASS:  Machine instructions     ← Hardware-level
```

---

## 2. Why Triton Matters for NVIDIA Interviews

1. **NVIDIA's own inference team uses Triton** for rapid kernel prototyping
2. FlashAttention-2's Triton implementation is widely used
3. Shows you can think at the GPU programming level without needing deep C++ CUDA
4. Many vLLM/SGLang custom kernels are written in Triton
5. It's the fastest path to "I can write GPU kernels" credibility

---

## 3. Triton vs CUDA — Key Differences

| Aspect | CUDA | Triton |
|---|---|---|
| Language | C/C++ | Python |
| Programming unit | Thread | Block (tile) |
| Memory management | Manual (shared mem, registers) | Automatic (compiler decides) |
| Synchronization | Manual (__syncthreads) | Automatic |
| Memory coalescing | Manual | Automatic |
| Bank conflict avoidance | Manual (padding) | Automatic |
| Learning curve | Steep (weeks) | Moderate (days) |
| Performance ceiling | Highest (full control) | ~90-95% of CUDA |
| Compilation | nvcc (ahead of time) | JIT (just in time) |

### The key insight
Triton's programming model is **block-level**, not thread-level:
- In CUDA: you think about what ONE thread does
- In Triton: you think about what a BLOCK of data does
- Triton compiler handles thread mapping, memory access patterns, synchronization

---

## 4. Triton Programming Model

### Block pointers
Instead of computing per-thread indices, Triton works with block pointers:

```python
# CUDA thinking (per-thread):
i = blockIdx.x * blockDim.x + threadIdx.x
c[i] = a[i] + b[i]

# Triton thinking (per-block):
offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
a_block = tl.load(a_ptr + offsets)
b_block = tl.load(b_ptr + offsets)
tl.store(c_ptr + offsets, a_block + b_block)
```

### Key Triton primitives
- `tl.program_id(axis)` — equivalent to blockIdx in CUDA
- `tl.arange(start, end)` — creates a range of offsets (like thread indices)
- `tl.load(pointer)` — loads a block of data from global memory
- `tl.store(pointer, value)` — stores a block of data to global memory
- `tl.dot(a, b)` — block-level matrix multiply (uses Tensor Cores automatically)
- `tl.max(x, axis)` — block-level reduction
- `tl.exp(x)` — element-wise operations on blocks
- `tl.where(cond, x, y)` — masked operations

### Auto-tuning
Triton supports automatic parameter tuning:
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}),
        # ... more configs
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...)
```
Triton benchmarks each config and picks the fastest for your hardware.

---

## 5. Vector Add in Triton

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Which block am I?
    pid = tl.program_id(axis=0)

    # Compute offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for out-of-bounds elements
    mask = offsets < n_elements

    # Load, compute, store
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# Launch
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output
```

### Compare to CUDA vector add:
- No manual `cudaMalloc`, `cudaMemcpy` — PyTorch handles memory
- No manual thread indexing — Triton works at block level
- Mask handles boundary conditions (like `if (i < n)` in CUDA)
- JIT compiled — writes kernel in Python, runs at GPU speed

---

## 6. Matrix Multiply in Triton

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Which output block am I computing?
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this block's rows and columns
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers to first block of A and B
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)  # Tensor Core matmul!
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc)
```

### Key observations:
- `tl.dot(a, b)` automatically uses Tensor Cores — you don't manage them
- The tiling loop is explicit (like CUDA tiled matmul) but shared memory is implicit
- Triton compiler handles shared memory allocation, bank conflict avoidance
- This achieves ~90% of cuBLAS performance with ~50 lines of Python

---

## 7. Fused Softmax in Triton

This is directly relevant to attention computation:

```python
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Load one row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(input_ptr + row_idx * input_row_stride + col_offsets, mask=mask, other=-float('inf'))

    # Compute softmax (numerically stable)
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(output_ptr + row_idx * output_row_stride + col_offsets, softmax_output, mask=mask)
```

### Why fused softmax matters:
Standard PyTorch softmax: 3 kernel launches (max, subtract+exp, sum+divide) = 3 HBM round trips.
Fused Triton softmax: 1 kernel = 1 HBM round trip. **3x fewer memory accesses.**

This is the same principle behind FlashAttention — fuse operations to minimize HBM traffic.

---

## 8. Causal Attention in Triton (Conceptual)

The stretch goal for hands-on: implement causal attention.

```python
# Pseudocode for causal attention in Triton:

@triton.jit
def causal_attention_kernel(Q, K, V, Out, ...):
    # Block of Q rows I'm responsible for
    q_block = load Q[block_m]

    # Initialize accumulator and softmax stats
    m_i = -inf    # running max
    l_i = 0.0     # running sum (softmax denominator)
    acc = zeros   # output accumulator

    # Loop over K,V blocks (only up to causal boundary!)
    for block_n in range(0, block_m + 1):  # Causal: don't look ahead
        k_block = load K[block_n]
        v_block = load V[block_n]

        # Compute attention scores
        s = tl.dot(q_block, tl.trans(k_block))

        # Apply causal mask within boundary block
        if block_n == block_m:
            causal_mask = (row_indices[:, None] >= col_indices[None, :])
            s = tl.where(causal_mask, s, -inf)

        # Online softmax update
        m_new = max(m_i, max(s))
        alpha = exp(m_i - m_new)
        p = exp(s - m_new)
        l_i = l_i * alpha + sum(p)
        acc = acc * alpha + tl.dot(p, v_block)
        m_i = m_new

    # Final normalization
    Out[block_m] = acc / l_i
```

This is essentially FlashAttention in Triton. Understanding this pseudocode means you understand both FlashAttention AND Triton.

---

## 9. Triton in the Ecosystem

### Where Triton kernels are used in production
- **FlashAttention:** Triton implementation available (alongside CUDA version)
- **vLLM:** Some custom kernels in Triton (attention variants, fused operations)
- **SGLang:** Uses Triton kernels for certain operations
- **xformers (Meta):** Memory-efficient attention uses Triton
- **Unsloth:** Fast LoRA fine-tuning uses custom Triton kernels
- **torch.compile:** PyTorch's compiler can generate Triton kernels automatically

### Triton vs custom CUDA kernels
- **Use Triton when:** Prototyping, moderate performance is OK, Python team
- **Use CUDA when:** Maximum performance needed, complex memory patterns, production at scale
- **In practice:** Start with Triton, only drop to CUDA if Triton's ~5-10% performance gap matters

---

## 10. Interview Connections

> "Have you written GPU kernels? Tell me about your experience."

**A:** "I've written kernels in both CUDA and Triton. In CUDA, I implemented vector add, naive and tiled matrix multiply, and softmax — this taught me GPU memory hierarchy, shared memory tiling, and the roofline model. In Triton, I implemented fused softmax and a causal attention kernel similar to FlashAttention — Triton's block-level programming model made it practical to implement tiled attention with online softmax in ~100 lines of Python while still achieving ~90% of CUDA performance."

> "What's the difference between CUDA and Triton? When would you use each?"

> "How does Triton achieve good performance without manual shared memory management?"

**A:** "Triton's compiler analyzes the block-level program and automatically handles shared memory allocation, memory coalescing, bank conflict avoidance, and thread synchronization. The programmer specifies WHAT blocks of data to load and compute, the compiler decides HOW to map that to threads and memory. This is similar to how a SQL optimizer handles query planning — you specify intent, not execution."

---

## Self-Test Questions

1. What is the programming unit in Triton vs CUDA?
2. How does `tl.dot()` differ from manual matmul in CUDA?
3. Why is fused softmax faster than PyTorch's standard softmax?
4. What does the Triton compiler automatically handle that CUDA requires manually?
5. Write pseudocode for vector add in Triton (from memory).
6. What is the significance of the causal mask in the attention kernel?
7. When would you choose CUDA over Triton?

---

## Notes
```


```
