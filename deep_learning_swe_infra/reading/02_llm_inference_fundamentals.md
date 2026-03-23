# LLM Inference Fundamentals

**Priority:** Read after GPU Architecture.
**Interview map:** Track A Round 2, Track B Rounds 2 & 3

---

## 1. The Two Phases of LLM Inference

Every LLM request has two phases. Understanding this distinction is fundamental.

### Prefill Phase (Processing the prompt)
- Takes the entire input prompt and processes it in one forward pass
- All tokens attend to each other simultaneously (matrix-matrix multiply)
- **Compute-bound** — GPU Tensor Cores are the bottleneck
- Output: KV cache entries for all input tokens + first output token
- This phase determines **TTFT (Time to First Token)**

### Decode Phase (Generating output tokens)
- Generates one token at a time, autoregressively
- Each new token attends to all previous tokens (matrix-vector multiply)
- **Memory-bandwidth-bound** — loading model weights from HBM is the bottleneck
- Output: one new token per step
- This phase determines **TPOT (Time Per Output Token)**

```
User prompt: "Explain GPU memory hierarchy"
                    │
            ┌───────▼───────┐
            │   PREFILL     │  Process all input tokens at once
            │   (compute    │  Matrix-matrix multiply
            │    bound)     │  Outputs: KV cache + first token
            └───────┬───────┘
                    │  ← TTFT measured here
            ┌───────▼───────┐
            │   DECODE      │  Generate tokens one-by-one
            │   (memory     │  Matrix-vector multiply
            │    bound)     │  Each step: load all weights from HBM
            └───────┬───────┘
                    │  ← TPOT measured per token
                    ▼
            "GPU memory hierarchy has four levels..."
```

---

## 2. KV Cache — Why It Exists and Why It's a Problem

### The Problem Without KV Cache
In a transformer, each new token must attend to ALL previous tokens. Without caching, you'd recompute the Key and Value projections for every previous token at every step. For a 2048-token sequence, generating the last token would require recomputing attention for all 2048 tokens. This is O(n²) per token generated.

### The Solution: KV Cache
Store the Key and Value projections for all previously processed tokens. When generating a new token, only compute Q/K/V for the NEW token, then attend to cached K/V from all previous tokens.

**Trade-off:** Saves massive computation but uses GPU memory.

### KV Cache Size Calculation
```
KV cache size per token = 2 × num_layers × num_heads × head_dim × dtype_size

Llama-3 8B (32 layers, 32 heads, 128 head_dim, FP16):
= 2 × 32 × 32 × 128 × 2 bytes = 524 KB per token

For 2048 tokens: 524 KB × 2048 = ~1 GB per request
For 100 concurrent requests: ~100 GB just for KV cache
```

### Why KV Cache Management Matters
- A100 has 80 GB of HBM
- Model weights (Llama-3 8B FP16) take ~16 GB
- That leaves ~64 GB for KV cache
- At 1 GB per request × 2048 context length → only ~64 concurrent requests
- Longer contexts = fewer concurrent requests = lower throughput
- This is why PagedAttention, quantized KV cache, and KV cache eviction strategies are critical research areas

### Interview question this answers:
> "What's KV cache and why does it fill up? How would you manage it for a multi-tenant system?"

---

## 3. TTFT and TPOT — The Metrics That Matter

### TTFT (Time to First Token)
- Time from request received to first token generated
- Dominated by prefill time
- User-perceived "responsiveness"
- Affected by: prompt length, model size, GPU compute speed, queue wait time
- **Typical targets:** <500ms for interactive, <2s for batch

### TPOT (Time Per Output Token)
- Time between consecutive output tokens
- Dominated by decode time (weight loading from HBM)
- User-perceived "streaming speed"
- Affected by: model size, GPU memory bandwidth, batch size, KV cache size
- **Typical targets:** <50ms for real-time, <100ms for interactive

### Throughput
- Total tokens generated per second across all requests
- = concurrent_requests / TPOT
- This is what matters for cost efficiency
- Batching improves throughput without improving individual TTFT

### End-to-End Latency
```
Total latency = TTFT + (num_output_tokens × TPOT)
```

### The Latency-Throughput Tradeoff
- Smaller batch → lower latency per request, lower throughput
- Larger batch → higher latency per request (more queue time), higher throughput
- The art of LLM serving = finding the right batch size for your SLO

### Interview question this answers:
> "A model's TPOT is 80ms and the customer needs 40ms. What do you look at first?"

**Answer framework:**
1. Memory bandwidth — are you saturating HBM? (nvidia-smi, Nsight)
2. Model size — can you quantize (FP16 → INT8/FP8) to halve weight loading time?
3. Tensor parallelism — split model across 2 GPUs, halve weights per GPU
4. KV cache — is it competing for HBM bandwidth? Can you use quantized KV cache?
5. Batch size — are you over-batching, causing shared bandwidth contention?

---

## 4. Continuous Batching (Iteration-Level Scheduling)

### The Problem with Static Batching
Traditional batching: collect N requests, run them together, wait for ALL to finish before returning any.

Problem: requests have different output lengths. A 10-token response waits for a 500-token response. GPU sits idle for finished requests.

```
Static Batching:
Request A (10 tokens):  [████]___________________  ← waiting, GPU wasted
Request B (50 tokens):  [████████████████████████]
Request C (20 tokens):  [████████]_______________  ← waiting, GPU wasted
                        All return together ────────►
```

### The Solution: Continuous Batching (from Orca paper)
Instead of batching at the request level, batch at the **iteration (token) level**.

After each decode step:
- Remove completed requests from the batch
- Add new requests from the queue
- Every decode step has a full batch → GPU is always busy

```
Continuous Batching:
Request A: [████] done → slot freed
Request D:       [████████████] ← takes A's slot immediately
Request B: [████████████████████████]
Request C: [████████] done → slot freed
Request E:            [██████████████] ← takes C's slot
```

### Why it matters:
- 2-3x throughput improvement over static batching
- Lower average latency (short requests don't wait for long ones)
- This is what vLLM, SGLang, TRT-LLM all implement
- NVIDIA NIM uses this under the hood

### In-Flight Batching (TRT-LLM term)
Same concept as continuous batching. TRT-LLM calls it "in-flight batching." It's the same thing.

### Interview question this answers:
> "How does continuous batching work and why does it matter for throughput?"

---

## 5. The Full Request Path

Understand this end-to-end. They WILL ask you to walk through it.

```
1. HTTP Request arrives
   └─► API Gateway / Load Balancer

2. Request Queue
   └─► Priority queue, rate limiting, SLO-based scheduling

3. Scheduler (the brain)
   ├─► Decides which requests to batch together
   ├─► Manages KV cache memory allocation (PagedAttention)
   ├─► Implements continuous batching (add/remove requests per iteration)
   └─► Handles preemption (pause low-priority requests if memory full)

4. Prefill
   ├─► Tokenize input
   ├─► Run full forward pass on input tokens (matrix-matrix multiply)
   ├─► Store KV cache for all input tokens
   └─► Output: first token + KV cache entries

5. Decode Loop (repeat until EOS or max_tokens)
   ├─► Run forward pass for ONE new token (matrix-vector multiply)
   ├─► Attention: new token Q attends to all cached K,V
   ├─► Append new K,V to cache
   ├─► Sample next token
   └─► Stream token back to client

6. Cleanup
   ├─► Free KV cache blocks
   ├─► Remove request from batch
   └─► Scheduler fills the slot with next queued request
```

### Interview question this answers:
> "Walk me through what happens when a token is generated — from the scheduler through attention to GPU execution"

---

## 6. Model Serving Architecture

For Track A (SA) — this is the systems design question.

```
┌─────────────────────────────────────────────────────┐
│                    Clients                           │
└────────────────────┬────────────────────────────────┘
                     │ HTTPS
┌────────────────────▼────────────────────────────────┐
│              API Gateway / Load Balancer             │
│         (Route53 + NLB/ALB or Kubernetes Ingress)   │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              Request Router                          │
│    ├─ Model routing (which model serves this?)      │
│    ├─ Rate limiting                                 │
│    ├─ Authentication / API key validation            │
│    └─ PII scrubbing (for regulated environments)    │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│           Inference Engine (vLLM / TRT-LLM / NIM)   │
│    ├─ Continuous batching scheduler                  │
│    ├─ KV cache manager (PagedAttention)              │
│    ├─ Tensor parallel execution (multi-GPU)          │
│    └─ Token streaming                                │
├─────────────────────────────────────────────────────┤
│               GPU Cluster                            │
│    ├─ Node 1: 8x A100 (NVLink interconnect)         │
│    ├─ Node 2: 8x A100                               │
│    └─ InfiniBand between nodes                      │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              Observability                           │
│    ├─ Prometheus metrics (TTFT, TPOT, throughput)    │
│    ├─ Grafana dashboards                             │
│    ├─ Audit logging (compliance)                     │
│    └─ Alerting on SLO violations                    │
└─────────────────────────────────────────────────────┘
```

### Key design decisions to discuss in interview:
1. **Multi-tenancy:** How to isolate KV cache and compute between tenants?
2. **Autoscaling:** Scale on queue depth, not CPU — GPU utilization is what matters
3. **Model versioning:** Blue/green deployment, canary rollouts
4. **Failover:** What happens when a GPU node dies mid-inference?
5. **Cost:** GPU-hours are expensive — optimize utilization (batching, right-sizing)

---

## Self-Test Questions

1. What are the two phases of LLM inference? Which is compute-bound and which is memory-bound?
2. Calculate KV cache size for Llama-3 70B (80 layers, 64 heads, 128 head_dim, FP16) at 4096 context length.
3. Explain continuous batching vs static batching. Why is continuous batching better?
4. What is TTFT? What is TPOT? Which phase dominates each?
5. Walk through the full request path from HTTP request to generated token.
6. A customer's TTFT is 2 seconds. What are 3 things you'd investigate?
7. Why does the KV cache create a tradeoff between context length and concurrency?

---

## Notes
```
(Take your own notes here)


```
