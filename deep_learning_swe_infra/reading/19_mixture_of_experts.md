# Mixture of Experts (MoE) Inference

**Priority:** Week 5-6.
**Interview map:** Track B Round 3, Track A Round 2

---

## 1. What is MoE?

A model architecture where each token is processed by only a SUBSET of the model's parameters.

```
Standard transformer (dense):
  Every token → ALL FFN parameters → output
  Llama-3 70B: 70B params active per token

MoE transformer (sparse):
  Every token → Router → selects top-K experts → only K experts process token
  Mixtral 8x7B: 8 experts (7B each) but only 2 active per token
  Total params: ~47B, Active params: ~13B per token
```

### Why MoE exists
- Get the quality of a large model (many parameters = more knowledge)
- At the cost of a small model (few active parameters = fast inference)
- Mixtral 8x7B matches Llama-2 70B quality but runs at ~13B model speed

---

## 2. Architecture

```
┌─────────────────────────────────────────┐
│         Standard Transformer Layer       │
│                                         │
│  Attention → LayerNorm → FFN → LayerNorm │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│           MoE Transformer Layer          │
│                                         │
│  Attention → LayerNorm → Router ─┐      │
│                           │      │      │
│                    ┌──────┴──────┐│     │
│                    │   top-K     ││     │
│                    │  selection  ││     │
│                    └──────┬──────┘│     │
│                           │      │      │
│              ┌────┬────┬──┴─┬────┤      │
│              │ E0 │ E1 │ E2 │ ...│      │
│              │(7B)│(7B)│(7B)│    │      │
│              └──┬─┴──┬─┴────┴────┘      │
│                 │    │                   │
│          Weighted sum of selected experts │
│                    ↓                     │
│               LayerNorm                  │
└─────────────────────────────────────────┘
```

### Router
- Small linear layer: `hidden_dim → num_experts`
- Outputs a score for each expert per token
- Top-K experts (usually K=2) are selected
- Selected expert outputs are weighted by router scores and summed

---

## 3. MoE Models

| Model | Experts | Active | Total Params | Active Params | Quality |
|---|---|---|---|---|---|
| Mixtral 8x7B | 8 | 2 | 46.7B | ~12.9B | ≈ Llama-2 70B |
| Mixtral 8x22B | 8 | 2 | 141B | ~39B | ≈ Llama-3 70B |
| DeepSeek-V2 | 160 | 6 | 236B | ~21B | Strong |
| DeepSeek-V3 | 256 | 8 | 671B | ~37B | ≈ GPT-4 class |
| DBRX | 16 | 4 | 132B | ~36B | Competitive |
| Grok-1 | 8 | 2 | 314B | ~86B | Strong |

### DeepSeek's approach
DeepSeek uses "fine-grained experts" — many small experts (160-256) instead of few large ones. This gives better load balancing and more flexible routing.

---

## 4. MoE Inference Challenges

### Challenge 1: Memory — Must load ALL experts
Even though only K experts are active per token, ALL expert weights must be in GPU memory (you don't know which will be selected until runtime).

```
Mixtral 8x7B: 46.7B params × 2 bytes (FP16) = ~93 GB
Active compute: only ~26 GB of params used per token
But need 93 GB in memory → still need 2x A100 80GB
```

### Challenge 2: Load Balancing
If the router sends most tokens to the same expert:
- That expert's GPU is overloaded
- Other experts' GPUs are idle
- Throughput collapses

**Solutions:**
- **Auxiliary load balancing loss** during training (penalize uneven routing)
- **Expert capacity factor:** cap how many tokens each expert processes
- **Token dropping:** if an expert is full, drop the token or route to second choice

### Challenge 3: All-to-All Communication
With expert parallelism (experts on different GPUs), tokens must be routed to the correct GPU:
```
Token on GPU 0 → needs Expert 3 on GPU 1 → send token to GPU 1
                → also needs Expert 5 on GPU 2 → send token to GPU 2
```
This is an **all-to-all communication** pattern — every GPU may need to send to every other GPU. More expensive than all-reduce (which has known patterns).

### Challenge 4: Batch Size Interaction
For dense models, larger batch = better GPU utilization.
For MoE, each expert sees only a fraction of the batch:
```
Batch size 64, 8 experts, top-2:
Each expert processes ~16 tokens on average (64 × 2 / 8)
But with imbalanced routing, some see 30 and others see 2
Small effective batch per expert → poor GPU utilization
```

---

## 5. MoE Parallelism Strategies

### Expert Parallelism (EP)
Assign different experts to different GPUs.
```
8 experts, 4 GPUs:
GPU 0: Expert 0, Expert 1
GPU 1: Expert 2, Expert 3
GPU 2: Expert 4, Expert 5
GPU 3: Expert 6, Expert 7

Communication: all-to-all to route tokens to correct GPU
```

### EP + TP
Combine expert parallelism with tensor parallelism:
```
8 experts, 8 GPUs:
Option A (EP=8): Each GPU holds 1 expert → all-to-all between all GPUs
Option B (EP=4, TP=2): 4 expert groups, each expert split across 2 GPUs
  GPU 0,1: Expert 0 (tensor parallel)
  GPU 2,3: Expert 1 (tensor parallel)
  ...
```

### EP + PP
For very large MoE models (DeepSeek-V3 at 671B):
```
Pipeline stages, each stage has its own set of experts
EP within each stage for expert distribution
```

---

## 6. MoE Serving Optimization

### Expert Offloading
When GPU memory is insufficient for all experts:
- Keep active (hot) experts in GPU memory
- Offload cold experts to CPU memory
- Load experts on-demand when routed
- Works because only K experts needed per token

**Trade-off:** CPU→GPU transfer adds latency. Works better with predictable routing patterns.

### Expert Caching
If routing patterns are predictable (e.g., certain prompts always route to certain experts):
- Cache recently used experts in GPU memory
- LRU eviction for less-used experts
- Can dramatically reduce offloading overhead

### Quantization for MoE
Quantization is especially valuable for MoE because:
- Memory is the bottleneck (all experts must fit)
- INT4 quantization: 93 GB → 23 GB for Mixtral 8x7B → fits on single GPU!
- Active compute is already small → quality impact is manageable

---

## 7. Interview Connections

> "How does MoE inference differ from dense model inference?"

**A:** "Three key differences: (1) Memory — all expert weights must be in memory even though only K are active per token. Mixtral 8x7B needs 93 GB despite only using 13B params per token. (2) Communication — tokens must be routed to the correct expert's GPU via all-to-all communication, which is more complex than TP's all-reduce. (3) Load balancing — uneven routing creates hot-spot experts and idle GPUs. The upside: compute per token is much less than a dense model of the same total size."

> "A customer wants to serve Mixtral 8x22B. What GPU configuration do you recommend?"

**A:** "Mixtral 8x22B is ~282 GB in FP16. With INT8 quantization: ~141 GB → fits on 2x A100 80GB. Use EP=2 (4 experts per GPU) or TP=2 (each expert split across GPUs). EP is simpler but requires all-to-all. For lowest latency, INT4 quantization: ~71 GB → fits on 1x A100 80GB, no communication overhead."

---

## Self-Test Questions

1. Why does MoE use more memory than its active parameter count suggests?
2. What is the router and how does it select experts?
3. What is expert parallelism? How does its communication pattern differ from tensor parallelism?
4. Why is load balancing a problem for MoE inference?
5. How does quantization help MoE models more than dense models?
6. Mixtral 8x7B has 46.7B total params. How many are active per token? Why?

---

## Notes
```


```
