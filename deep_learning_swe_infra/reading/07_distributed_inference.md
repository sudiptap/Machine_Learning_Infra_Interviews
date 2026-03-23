# Distributed Inference — Tensor and Pipeline Parallelism

**Priority:** Read in Week 3–4.
**Interview map:** Track A Round 4, Track B Rounds 2 & 3

---

## 1. Why Distributed Inference?

Some models are too large for a single GPU:

| Model | FP16 Size | GPUs Needed (A100 80GB) |
|---|---|---|
| Llama-3 8B | ~16 GB | 1 |
| Llama-3 70B | ~140 GB | 2 |
| Llama-3 405B | ~810 GB | 11+ (typically 16) |

Even when a model fits on one GPU, distributing it can improve latency (each GPU does less work per token).

---

## 2. Tensor Parallelism (TP)

### What it is
Split individual layers **horizontally** across GPUs. Each GPU holds a portion of every layer.

### How it works for a transformer:

**Attention heads:**
- A model with 32 attention heads on 4 GPUs → 8 heads per GPU
- Each GPU computes attention for its heads independently
- Results are combined via **all-reduce** (sum across GPUs)

**MLP (Feed-Forward):**
- Weight matrix is split column-wise or row-wise across GPUs
- Each GPU computes its portion of the matmul
- Combined via **all-reduce**

```
TP=4 (4 GPUs):

Layer N:
GPU 0: [heads 0-7]  [MLP shard 0]  ─┐
GPU 1: [heads 8-15] [MLP shard 1]  ─┤── all-reduce after each layer
GPU 2: [heads 16-23][MLP shard 2]  ─┤
GPU 3: [heads 24-31][MLP shard 3]  ─┘

Layer N+1:
GPU 0: [heads 0-7]  [MLP shard 0]  ─┐
...same pattern...                    ─┘
```

### Communication overhead
- **2 all-reduce operations per transformer layer** (one after attention, one after MLP)
- All-reduce sends O(hidden_dim) data between all GPUs
- Requires **high-bandwidth interconnect** (NVLink, not PCIe)

### When to use
- Model fits on N GPUs but not 1
- Need lowest possible latency (all GPUs work in parallel on same token)
- Have NVLink interconnect (TP over PCIe is too slow)

### Constraints
- TP degree must divide the number of attention heads evenly
- TP within a single node (8 GPUs with NVLink) — don't do TP across nodes (InfiniBand too slow for per-layer communication)

---

## 3. Pipeline Parallelism (PP)

### What it is
Split the model **vertically** — different layers on different GPUs. Data flows through GPUs sequentially like a pipeline.

```
PP=4 (4 GPUs):

GPU 0: [Layers 0-7]   ──►
GPU 1: [Layers 8-15]  ──►
GPU 2: [Layers 16-23] ──►
GPU 3: [Layers 24-31] ──►
```

### How it works
1. Request enters GPU 0, processes through layers 0-7
2. Output activations sent to GPU 1 (point-to-point, not all-reduce)
3. GPU 1 processes layers 8-15, sends to GPU 2
4. ... and so on

### The bubble problem
With a single request, only one GPU is active at a time — the others wait:

```
Time →
GPU 0: [████]
GPU 1:       [████]
GPU 2:             [████]
GPU 3:                   [████]
                                  ← 75% of GPU time is wasted (bubble)
```

### Microbatching reduces the bubble
Split the batch into microbatches. While GPU 1 processes microbatch 1, GPU 0 can process microbatch 2:

```
Time →
GPU 0: [mb1][mb2][mb3][mb4]
GPU 1:      [mb1][mb2][mb3][mb4]
GPU 2:           [mb1][mb2][mb3][mb4]
GPU 3:                [mb1][mb2][mb3][mb4]
                                          ← Less wasted time
```

### Communication overhead
- **Point-to-point communication** between adjacent GPUs only
- Much less data than TP's all-reduce
- Works over InfiniBand (lower bandwidth OK since it's point-to-point)

### When to use
- Model is too large even for TP across all GPUs in a node
- Multi-node deployment (InfiniBand between nodes)
- High-throughput scenarios where microbatching fills the pipeline

---

## 4. Combining TP and PP

For very large models, use both:

```
Example: Llama-3 405B on 2 nodes, 8 GPUs each (16 total)

Node 0 (NVLink within):
  TP=8: All 8 GPUs share layers 0-39 (tensor parallel)

Node 1 (NVLink within):
  TP=8: All 8 GPUs share layers 40-79 (tensor parallel)

PP=2: Node 0 → Node 1 (pipeline parallel over InfiniBand)
```

**Rule of thumb:**
- **TP within a node** (needs NVLink bandwidth)
- **PP across nodes** (tolerates InfiniBand latency)

---

## 5. Sequence Parallelism (SP)

### What it is
During TP, operations like LayerNorm and Dropout are replicated on every GPU (same computation, wasted). Sequence parallelism splits the **sequence dimension** for these operations.

### How it works
- For attention + MLP: use tensor parallelism (split heads/weights)
- For LayerNorm, Dropout: split along sequence dimension across GPUs
- Reduces redundant computation and memory usage

### When it matters
- Very long sequences where the per-GPU memory for activations is significant
- Megatron-LM implements this

---

## 6. Expert Parallelism (for MoE models)

### What it is
Mixture-of-Experts models (like Mixtral) have multiple "expert" FFN layers. Only a subset is activated per token. Expert Parallelism assigns different experts to different GPUs.

```
Mixtral 8x7B: 8 experts, each ~7B parameters

GPU 0: Expert 0, Expert 1
GPU 1: Expert 2, Expert 3
GPU 2: Expert 4, Expert 5
GPU 3: Expert 6, Expert 7

Router selects top-2 experts per token → tokens routed to correct GPUs
```

### Communication
- Requires **all-to-all** communication: any token might go to any expert
- Load balancing is critical — if all tokens route to the same expert, GPUs are unbalanced

---

## 7. NCCL — The Communication Library

### What it is
NVIDIA Collective Communication Library — implements efficient multi-GPU communication primitives:
- **All-reduce:** Sum tensors across all GPUs (used in TP)
- **All-gather:** Gather tensors from all GPUs to all GPUs
- **Reduce-scatter:** Reduce and distribute
- **Point-to-point send/recv:** Used in PP

### Why it matters
- All multi-GPU NVIDIA inference uses NCCL
- It's optimized for NVLink and InfiniBand topologies
- Performance of distributed inference is often limited by NCCL communication overhead

---

## 8. Practical Decision Framework

```
Model fits on 1 GPU?
  → Yes: No parallelism needed. Single GPU serving.
  → No: Continue.

Model fits on GPUs within 1 node (e.g., 8x A100)?
  → Yes: Use Tensor Parallelism (TP=2, 4, or 8)
  → No: Continue.

Model needs multiple nodes?
  → Use TP within each node + PP across nodes
  → Example: 2 nodes × 8 GPUs = TP=8, PP=2
```

### Latency vs Throughput
- TP reduces latency (parallel computation per token)
- PP can increase throughput (pipeline different requests)
- TP=2 roughly halves TPOT (less weights per GPU to load)

---

## Interview Questions This Answers

> "What's the difference between tensor parallelism and pipeline parallelism? When would you choose each?"

**Answer:** "Tensor parallelism splits each layer across GPUs — every GPU processes every token but on a subset of the weights. It requires high-bandwidth interconnect (NVLink) because there's an all-reduce after every layer. Pipeline parallelism splits layers across GPUs — each GPU processes the full width of its layers but only a portion of the depth. It only needs point-to-point communication between adjacent stages, so it works over InfiniBand. Use TP within a node for latency, PP across nodes for scale."

> "A customer has an A100 80GB. They want to serve Llama-3 70B at low latency. Walk me through what limits them and how you'd configure tensor parallelism."

**Answer:** "Llama-3 70B in FP16 is ~140 GB — won't fit on one A100 80GB. Minimum TP=2 (70 GB per GPU). With TP=2, each decode step loads ~70 GB of weights per GPU at 2 TB/s bandwidth = ~35ms minimum TPOT, plus all-reduce overhead (~1-2ms per layer over NVLink). For lower latency, TP=4 gives ~17.5ms TPOT theoretically. Alternatively, INT8 quantization reduces to ~70 GB total, fitting on 1 GPU — no TP overhead, ~35ms TPOT."

---

## Self-Test Questions

1. Draw a diagram of TP=4 for a transformer layer. Show which GPU holds what.
2. What communication primitive does TP use? PP?
3. Why should TP be within a node and PP across nodes?
4. What is the pipeline bubble and how do microbatches reduce it?
5. Llama-3 405B on 16 GPUs (2 nodes of 8). Design the TP/PP configuration.
6. When does TP reduce latency? When does it not help?
7. What is NCCL and what operations does it provide?

---

## Notes
```
(Take your own notes here)


```
