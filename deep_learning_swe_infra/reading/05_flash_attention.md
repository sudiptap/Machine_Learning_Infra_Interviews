# FlashAttention — The Core Insight

**Priority:** Read in Week 2–3.
**Interview map:** Track B Round 3 ("Why does FlashAttention reduce memory usage?")
**Papers:** FlashAttention (2022), FlashAttention-2 (2023)

---

## 1. The Problem: Standard Attention is Memory-Wasteful

Standard self-attention for sequence length N:

```python
# Standard attention (PyTorch-style)
Q, K, V = linear(x)           # Each is (batch, heads, seq_len, head_dim)
S = Q @ K.T                   # (batch, heads, seq_len, seq_len) — HUGE matrix
P = softmax(S)                # Same size — ANOTHER huge matrix
O = P @ V                     # (batch, heads, seq_len, head_dim)
```

### The memory problem:
- `S` and `P` are both N×N matrices
- For N=4096: each is 4096² × 2 bytes (FP16) = **32 MB per head per batch element**
- With 32 heads and batch 8: 32 × 8 × 32 MB = **8 GB** just for attention intermediates
- These matrices are **materialized in HBM** (slow GPU memory)
- They're read back from HBM for the softmax and V multiplication

### The speed problem:
- Writing S to HBM then reading it back for softmax = 2 HBM round trips
- Writing P to HBM then reading it back for P@V = 2 more HBM round trips
- These HBM accesses dominate runtime (attention is memory-bound for typical sizes)

---

## 2. FlashAttention — The Solution

### Core insight: Never materialize the N×N attention matrix.

Instead of computing S, P, O sequentially with full matrices in HBM:
- **Tile** the computation into small blocks that fit in **SRAM (shared memory)**
- Process one tile of Q against all tiles of K,V at a time
- Use **online softmax** to compute the correct softmax without seeing the full row

```
Standard:  Q ──► [S = QK^T in HBM] ──► [P = softmax(S) in HBM] ──► [O = PV in HBM]
                 ▲ write to HBM          ▲ write to HBM

Flash:     Q tiles ──► [compute QK^T in SRAM, running softmax, accumulate PV in SRAM] ──► O
                       ▲ never write N×N matrix to HBM
```

### Tiling Strategy

```
Q is divided into blocks of size Br (block rows)
K, V are divided into blocks of size Bc (block columns)

For each Q block:
  Initialize: O_block = 0, l = 0 (softmax denominator), m = -inf (running max)
  For each K, V block:
    1. Load Q_block, K_block, V_block into SRAM
    2. Compute S_block = Q_block @ K_block.T    (in SRAM, small matrix)
    3. Update running max: m_new = max(m, max(S_block))
    4. Compute P_block = exp(S_block - m_new)   (in SRAM)
    5. Update softmax denominator: l = l * exp(m - m_new) + sum(P_block)
    6. Update output: O_block = O_block * exp(m - m_new) + P_block @ V_block
    7. m = m_new
  Final: O_block = O_block / l    (correct softmax normalization)
```

### Why online softmax works:
Standard softmax requires the max of the entire row (for numerical stability):
```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

Online softmax maintains a **running max** and **running sum**. When a new block reveals a larger max, it rescales the previous partial results using `exp(old_max - new_max)`. The final result is mathematically identical.

---

## 3. Why FlashAttention Is Faster

### Memory complexity
| | Standard Attention | FlashAttention |
|---|---|---|
| HBM reads/writes | O(N² + Nd) | O(N²d² / M) |
| Extra HBM memory | O(N²) for S, P | O(N) for logsumexp |

Where M = SRAM size, N = seq length, d = head dimension.

### In practice:
- FlashAttention reduces HBM access by **5-20x** for typical sizes
- Since attention is memory-bound, fewer HBM accesses = directly faster
- **2-4x speedup** on A100 for typical LLM sequence lengths

### Memory savings:
- Standard: O(N²) extra memory → 32 MB per head at seq_len 4096
- Flash: O(N) extra memory → ~8 KB per head
- Enables much longer sequences on the same GPU

---

## 4. FlashAttention-2 Improvements

### Better parallelism
- FA1: parallelizes over batch size and number of heads
- FA2: ALSO parallelizes over sequence length (better GPU occupancy for long sequences)

### Better work partitioning
- FA2 reduces non-matmul FLOPs (the overhead of rescaling, max tracking)
- Better utilization of Tensor Cores (more time in matmul, less in scalar ops)

### Causal masking
- FA2 handles causal (autoregressive) masking efficiently
- Skips computation for masked blocks entirely (not just masking after computing)
- ~2x speedup for causal attention vs non-causal

### Results
- FA2 achieves **50-73% of theoretical maximum FLOPS** on A100
- Standard attention achieves **25-40%**

---

## 5. FlashAttention for Decode (FlashDecoding)

Standard FlashAttention is optimized for prefill (long Q sequence).

During decode, Q has only 1 token but K,V can be very long (full KV cache). Different optimization needed:

**FlashDecoding:** Parallelizes over KV sequence length rather than Q length.
- Split KV cache into chunks
- Each chunk computes partial attention in parallel
- Reduce partial results with online softmax correction
- Much better GPU utilization during decode

---

## 6. Connection to Other Concepts

### FlashAttention + PagedAttention (FlashInfer)
- FlashAttention assumes contiguous KV cache
- PagedAttention stores KV cache in non-contiguous blocks
- **FlashInfer** library implements FlashAttention-style tiling that works with paged KV cache
- This is what vLLM and SGLang use in practice

### FlashAttention + Tensor Parallelism
- With tensor parallelism, attention heads are split across GPUs
- Each GPU runs FlashAttention on its subset of heads
- All-reduce needed only for the MLP layers, not within attention

### FlashAttention + Quantization
- Can use FP8 or INT8 for Q, K, V to further reduce memory bandwidth
- H100 Tensor Cores natively support FP8 → FlashAttention with FP8 is even faster

---

## 7. The Interview Answer

> "Why does FlashAttention reduce memory usage? What's the core insight?"

**Your answer:**

"Standard attention materializes the full N×N attention matrix in HBM — that's O(N²) memory and requires multiple round trips to slow global memory. FlashAttention's core insight is to never materialize this matrix. Instead, it tiles the Q, K, V matrices into blocks that fit in SRAM (fast on-chip memory), computes attention tile-by-tile, and uses an online softmax algorithm to maintain correct normalization across tiles without needing the full attention matrix. This reduces HBM access by 5-20x and memory usage from O(N²) to O(N). Since attention is memory-bandwidth-bound for typical sizes, fewer HBM accesses directly translates to faster execution — 2-4x speedup in practice."

---

## Self-Test Questions

1. Why does standard attention require O(N²) memory? What are the two large matrices?
2. What is online softmax and why is it necessary for FlashAttention?
3. Draw the tiling pattern: how does FlashAttention iterate over Q blocks and K,V blocks?
4. Why is FlashAttention faster if it does the same number of FLOPs?
5. What changed between FlashAttention-1 and FlashAttention-2?
6. How does FlashDecoding differ from standard FlashAttention?
7. How do FlashAttention and PagedAttention work together?

---

## Notes
```
(Take your own notes here)


```
