# Megatron-LM — Large-Scale Model Parallelism

**Priority:** Week 5-6.
**Interview map:** Track B Round 2 (systems design), Track A Round 4
**Paper:** Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (2019, updated 2021)

---

## 1. What is Megatron-LM?

NVIDIA's framework for training and serving large language models with model parallelism. It introduced the parallelism strategies that ALL modern LLM systems now use.

Key contributions:
1. **Efficient tensor parallelism** for transformers
2. **Pipeline parallelism** with interleaved scheduling
3. **Sequence parallelism** to reduce activation memory
4. Showed how to scale to trillions of parameters

---

## 2. Megatron's Tensor Parallelism — The Details

### Splitting the MLP

A transformer MLP has two linear layers:
```
h → [Linear1: (h, 4h)] → GeLU → [Linear2: (4h, h)] → output
```

**Megatron's approach:**

Split Linear1 **column-wise** across GPUs:
```
GPU 0: A₁ = X × W₁[:, :2h]    → (batch, 2h)
GPU 1: A₂ = X × W₁[:, 2h:]    → (batch, 2h)
```
Each GPU applies GeLU to its portion independently (no communication needed).

Split Linear2 **row-wise** across GPUs:
```
GPU 0: Y₁ = GeLU(A₁) × W₂[:2h, :]
GPU 1: Y₂ = GeLU(A₂) × W₂[2h:, :]
Output = Y₁ + Y₂    → all-reduce
```

**Communication:** One all-reduce after MLP (sum partial results).

### Splitting Self-Attention

Split attention heads across GPUs:
```
4 GPUs, 32 heads → 8 heads per GPU

GPU 0: heads 0-7   → Q₀, K₀, V₀ → Attention₀ → O₀
GPU 1: heads 8-15  → Q₁, K₁, V₁ → Attention₁ → O₁
GPU 2: heads 16-23 → ...
GPU 3: heads 24-31 → ...

After attention: O = concat(O₀, O₁, O₂, O₃) × W_out
W_out is split row-wise → all-reduce
```

**Communication:** One all-reduce after attention output projection.

### Total communication per transformer layer
- **2 all-reduces:** one after attention, one after MLP
- Each all-reduce sends `batch_size × seq_len × hidden_dim × dtype_size` bytes
- For Llama-3 70B (hidden=8192, FP16): 2 × batch × seq × 8192 × 2 bytes per layer

### Why this is efficient
- All-reduces are the only communication — computation is fully parallel
- With NVLink (900 GB/s bidirectional on H100), all-reduce for 8192×2 bytes is microseconds
- Computation (matmul) takes milliseconds → communication is <1% of total time

---

## 3. Pipeline Parallelism — Interleaved Schedule

### The bubble problem (review)
With naive PP, GPUs sit idle during startup and teardown:
```
4 stages, 1 microbatch:
Stage 0: [F][ ][ ][ ][B][ ][ ][ ]
Stage 1: [ ][F][ ][ ][ ][B][ ][ ]
Stage 2: [ ][ ][F][ ][ ][ ][B][ ]
Stage 3: [ ][ ][ ][F][ ][ ][ ][B]
Bubble = (P-1)/P of total time  (P = number of stages)
```

### 1F1B Schedule (One Forward One Backward)
Megatron's first improvement — interleave forward and backward passes:
```
Stage 0: [F₁][F₂][F₃][F₄][B₁][B₂][B₃][B₄]
Stage 1:     [F₁][F₂][F₃][B₁][B₂][B₃][F₄][B₄]
(simplified — actual interleaving is more complex)
```
Bubble reduced to `(P-1) / (P-1+M)` where M = microbatches.

### Interleaved Pipeline
Assign **non-contiguous** layers to each stage:
```
Instead of: Stage 0 = layers 0-9, Stage 1 = layers 10-19, ...
Do:         Stage 0 = layers 0-4, 20-24
            Stage 1 = layers 5-9, 25-29
            Stage 2 = layers 10-14, 30-34
            Stage 3 = layers 15-19, 35-39
```
Each stage processes smaller chunks → more microbatches fit → smaller bubble.
Bubble reduced by ~50% compared to non-interleaved.

**Trade-off:** More point-to-point communication (stages talk more frequently).

---

## 4. Sequence Parallelism

### The problem
With tensor parallelism, operations like LayerNorm and Dropout are **replicated** on every GPU — same computation, same memory, wasted.

### The solution
Split the sequence dimension for non-tensor-parallel operations:
```
Tensor Parallel regions (attention, MLP):
  Split along head/hidden dimension — each GPU has different data

Non-TP regions (LayerNorm, Dropout):
  Split along sequence dimension — each GPU processes different tokens

Transition: all-gather before TP region, reduce-scatter after TP region
```

### Memory savings
- Activation memory for LayerNorm and Dropout reduced by TP_degree
- For TP=8: saves ~12.5% of total activation memory per GPU
- Becomes significant for very long sequences

---

## 5. The 3D Parallelism Strategy

For very large models, combine all three:

```
Example: 1 trillion parameter model on 512 GPUs (64 nodes × 8 GPUs)

Within each node (8 GPUs, NVLink):
  Tensor Parallelism = 8 (split each layer across all 8 GPUs)
  Sequence Parallelism = 8 (for non-TP ops)

Across nodes (64 nodes, InfiniBand):
  Pipeline Parallelism = 64 (each node handles ~1/64 of layers)

Total: TP=8, PP=64, DP=1 → 8 × 64 = 512 GPUs
```

### Adding Data Parallelism
If you have more GPUs than needed for model parallelism:
```
2048 GPUs, model needs 512 for TP×PP:
  TP=8, PP=64, DP=4
  4 copies of the model, each on 512 GPUs
  Data parallelism across the 4 copies
  → 4× throughput
```

### Decision framework
```
1. TP first: set TP = number of GPUs per node (usually 8)
   - Needs NVLink bandwidth
   - Reduces per-GPU memory and computation

2. PP next: set PP = number of nodes needed for model to fit
   - Needs InfiniBand (less bandwidth OK)
   - Each node holds a subset of layers

3. DP last: replicate across remaining GPUs for throughput
   - Only communication: gradient all-reduce (done asynchronously)
   - Linear throughput scaling
```

---

## 6. Megatron-LM for Inference

While Megatron-LM was designed for training, its parallelism strategies are used directly in inference:

### TensorRT-LLM
- Uses Megatron's TP strategy for splitting attention heads and MLP
- Uses Megatron's PP for multi-node inference
- The weight layout follows Megatron's sharding conventions

### vLLM
- Implements TP following Megatron's column/row-wise splitting
- Uses NCCL all-reduce (same as Megatron)

### NeMo Inference
- Directly loads Megatron-trained checkpoints
- Same parallelism configuration for inference as training

---

## 7. Key Formulas

### Communication volume per transformer layer (TP)
```
All-reduce volume = 2 × batch × seq × hidden × dtype_size
(factor of 2 for ring all-reduce: reduce-scatter + all-gather)

Two all-reduces per layer → 4 × batch × seq × hidden × dtype_size per layer
```

### Pipeline bubble fraction
```
Bubble = (P - 1) / (P - 1 + M)
where P = pipeline stages, M = microbatches
For P=4, M=32: bubble = 3/35 = 8.6%
For P=4, M=4:  bubble = 3/7 = 43% (much worse)
→ Need many microbatches to keep bubble small
```

### Memory per GPU with 3D parallelism
```
Model memory per GPU = total_model_params × dtype_size / (TP × PP)
Activation memory per GPU ≈ batch × seq × hidden × num_layers_per_stage × dtype_size / TP
KV cache per GPU = per_token_kv_size × max_seq_len × max_batch / TP
```

---

## 8. Interview Connections

> "How does Megatron-LM's tensor parallelism work for the MLP layer?"

**A:** "Megatron splits the first MLP linear layer column-wise and the second row-wise across GPUs. This means each GPU computes a portion of the intermediate activation independently, applies GeLU locally, then computes its portion of the output. The partial outputs are summed via all-reduce. This requires only one all-reduce per MLP — minimal communication."

> "You need to serve a 1T parameter model. How do you configure parallelism?"

**A:** "TP=8 within each node (NVLink needed for per-layer all-reduce). PP across nodes via InfiniBand. At 1T params FP16 = 2 TB, with TP=8 that's 250 GB per stage set — need ~4 pipeline stages → PP=4, so 32 GPUs minimum. For throughput, add DP on top. So: TP=8, PP=4, DP=N for 32×N GPUs."

> "What is the pipeline bubble and how do you minimize it?"

---

## Self-Test Questions

1. How does Megatron split the MLP across GPUs? Why column-wise first, row-wise second?
2. How many all-reduces per transformer layer with Megatron TP?
3. What is the pipeline bubble? Give the formula.
4. What is sequence parallelism and why does it help?
5. Design a parallelism strategy for Llama-3 405B on 2 nodes of 8 H100s.
6. Why is TP done within a node and PP across nodes?

---

## Notes
```


```
