# Emerging Research in LLM Inference

**Priority:** Week 10-12. Shows you're current, differentiates you in interviews.
**Interview map:** Track B Round 3 (shows depth), Track A Round 2 (shows awareness)

---

## 1. Ring Attention / Context Parallelism

### Problem
Long context (100K+ tokens) requires enormous KV cache that doesn't fit on one GPU.
Standard TP splits attention heads but each GPU still holds the full sequence.

### Solution: Ring Attention
Split the **sequence** across GPUs. Each GPU holds a chunk of KV cache.

```
GPU 0: tokens 0-25K       (KV cache for these tokens)
GPU 1: tokens 25K-50K     (KV cache for these tokens)
GPU 2: tokens 50K-75K     (KV cache for these tokens)
GPU 3: tokens 75K-100K    (KV cache for these tokens)

For each Q block:
  Pass K,V chunks around the ring, compute partial attention at each stop
  GPU 0 → sends K₀,V₀ to GPU 1
  GPU 1 → computes attention(Q₁, K₀) while receiving K₁ from GPU 2
  ... and so on in a ring pattern
```

### Key insight
Communication (sending K,V chunks) is **overlapped** with computation (attention on current chunk). If the chunk is large enough, computation dominates → near-linear scaling.

### Status
- Research papers from UC Berkeley, Google
- Used in training for extremely long contexts (1M+ tokens)
- Beginning to appear in inference for long-context models

### Interview angle
> "How would you serve a model with 1M context length?"

---

## 2. Tree Attention / Cascade Inference

### Problem
Speculative decoding generates a single draft sequence. But the draft model might have multiple plausible next tokens — exploring only one is wasteful.

### Solution: Tree-structured speculation
Generate a TREE of candidate sequences, verify the entire tree at once.

```
Standard speculative decoding:
  Draft: [A] → [B] → [C] → [D]   (linear chain)
  If B is rejected → wasted C and D

Tree speculation:
  Draft:     [A] → [B] → [C]
               ↘ [E] → [F]
                   ↘ [G]
  Verify all branches simultaneously
  Accept the longest valid branch
```

### Benefit
- Higher acceptance rates (more paths explored)
- Better utilization of the verification forward pass
- Can generate 3-5 tokens per forward pass instead of 1-2

### Implementations
- Sequoia (2024): optimal tree structure based on draft model statistics
- Medusa: multi-head approach (predicts future positions independently)

---

## 3. Disaggregated Serving at Scale

### Beyond prefill-decode disaggregation
Emerging architectures disaggregate MORE components:

```
Traditional: [Everything on GPU]

Current disaggregation:
  [Prefill GPUs] → KV transfer → [Decode GPUs]

Emerging disaggregation:
  [Prefill GPUs] → KV transfer → [Decode GPUs]
  [KV Cache Store (distributed memory pool)]
  [Attention GPUs (specialized for attention computation)]
  [MLP GPUs (specialized for feed-forward computation)]
```

### MemServe / Mooncake pattern
- Disaggregate KV cache into a shared memory pool
- Multiple prefill and decode workers access the same pool
- Enables cluster-level prefix caching (share KV across nodes)
- Dramatically improves cache hit rates for production workloads

### Why this matters
At scale (1000+ GPUs), disaggregation can reduce costs by 40-60% by specializing hardware for each phase and sharing KV cache across the cluster.

---

## 4. Inference-Time Compute Scaling

### The idea (from OpenAI o1/o3, DeepSeek-R1)
Instead of always generating a quick answer, let the model "think longer" on hard problems:
- Generate chain-of-thought reasoning
- Verify and retry if stuck
- Allocate more compute to harder problems

### Implications for serving
```
Traditional: fixed compute per request (~same TPOT for all)

Inference-time scaling:
  Easy question → 50 tokens, 2 seconds
  Hard question → 5000 tokens of reasoning, 200 seconds
  → 100x compute difference between requests!
```

### Challenges
- **Scheduling:** How to prioritize when some requests take 100x longer?
- **SLOs:** Can't promise <5s latency if model might think for 200s
- **Cost:** Customer pays per token — "thinking tokens" are expensive
- **Batching:** Long-running requests block batch slots

### Emerging patterns
- **Budget-constrained inference:** Limit thinking tokens per request based on priority/cost
- **Adaptive compute:** Start with fast answer, escalate to deep thinking if confidence is low
- **Parallel thinking:** Generate multiple reasoning paths, pick the best

---

## 5. Multimodal Inference

### Growing importance
- Vision-Language Models (VLMs): LLaVA, GPT-4V, Gemini
- Audio-Language Models: Whisper + LLM
- Video understanding: long video → token sequence → LLM

### Inference challenges
```
Text-only: input = tokens (small)
VLM: input = image (hundreds of tokens after encoding) + text

Image token overhead:
  1 image → ~576-2048 tokens (depends on resolution)
  Equivalent to a 2000-word prompt
  → Prefill is much heavier for multimodal
```

### Optimization strategies
- **Image token compression:** Reduce visual tokens while preserving information
- **Cached visual embeddings:** Cache encoded images for repeated queries
- **Dynamic resolution:** Smaller image tokens for simple queries, full resolution for detailed analysis
- **Separate visual encoder:** Run image encoder on CPU/separate GPU, only LLM on inference GPU

### Relevance
SGLang has strong multimodal support — your SGLang knowledge connects here.

---

## 6. Efficient Long-Context Techniques

### The problem
- KV cache grows linearly with context length
- Attention is O(n²) in standard implementation
- 100K context → massive KV cache and slow attention

### Solutions

**Linear Attention:**
- Replace softmax attention with linear operations
- O(n) instead of O(n²)
- Models: Mamba, RWKV, RetNet
- Trade-off: some quality loss for very long-range dependencies

**Hybrid Architectures:**
- Mix transformer layers (good at long-range) with linear layers (efficient)
- Jamba (AI21): interleaves Transformer + Mamba layers
- Gets benefits of both: quality from attention, efficiency from linear

**Sparse Attention:**
- Don't attend to all tokens — only attend to important ones
- Learned sparsity patterns (BigBird, Longformer)
- Works well when most attention mass is on a few key tokens

**Retrieval-Based:**
- Instead of long context, retrieve relevant chunks on demand
- KV cache stays small, relevance is maintained via retrieval
- This is essentially RAG as an alternative to long context

---

## 7. Hardware Trends

### H200 and B100/B200
- H200: 141 GB HBM3e (vs H100's 80 GB) → 76% more KV cache capacity
- B200: 192 GB HBM3e, NVLink 5.0 (1.8 TB/s), 2x H100 inference perf

### Implication for inference
- More memory = more concurrent requests = higher throughput
- Higher bandwidth = faster decode (load weights faster)
- Llama-3 70B in FP16 fits on a SINGLE B200 (192 GB) — no TP needed

### Custom inference chips
- Groq: LPU (Language Processing Unit) — deterministic latency, no batching needed
- Cerebras: Wafer-scale engine — extreme memory bandwidth
- SambaNova: Reconfigurable dataflow — optimized for specific models
- These are niche but worth knowing about for "what's the competitive landscape?" questions

---

## 8. Interview Connections

> "What's the latest research in LLM inference that excites you?"

Pick ONE topic and go deep:

**Option A (Ring Attention):**
"Ring Attention for long-context serving. As context lengths grow to 1M tokens, KV cache doesn't fit on one GPU even with quantization. Ring Attention splits the sequence across GPUs and passes K,V chunks in a ring, overlapping communication with computation. This enables near-linear scaling of context length with GPU count."

**Option B (Disaggregated serving):**
"Cluster-level KV cache disaggregation. Instead of per-GPU KV cache, systems like Mooncake use a distributed memory pool. This enables cluster-level prefix caching — when any node processes a prompt, its KV cache is available to all other nodes. At production scale with shared system prompts, this can eliminate 80%+ of prefill computation."

**Option C (Inference-time scaling):**
"Inference-time compute scaling, as seen in reasoning models. This fundamentally changes serving — some requests need 100x more compute than others. Traditional batching and scheduling assumptions break down. The serving system needs to adaptively allocate compute, possibly with budget-constrained inference where harder problems get more thinking tokens."

---

## Self-Test Questions

1. What is Ring Attention and when would you use it?
2. How does tree-structured speculative decoding improve over standard speculative decoding?
3. What is cluster-level KV cache disaggregation and why does it help at scale?
4. How does inference-time compute scaling (reasoning models) challenge traditional serving?
5. What is the KV cache implication of multimodal (vision-language) models?
6. Name 3 approaches to efficient long-context inference.
7. How much VRAM does B200 have? What does this change for serving?

---

## Notes
```


```
