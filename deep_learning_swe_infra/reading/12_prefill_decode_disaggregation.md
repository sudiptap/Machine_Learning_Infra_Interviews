# Prefill-Decode Disaggregation

**Priority:** Week 2-3. Cutting-edge architecture pattern.
**Interview map:** Track B Round 2 (systems design), Track A Round 4

---

## 1. The Problem: Prefill and Decode Compete

Prefill and decode have fundamentally different compute profiles:

| | Prefill | Decode |
|---|---|---|
| Bound by | Compute (Tensor Cores) | Memory bandwidth (HBM) |
| GPU utilization | High (dense matmul) | Low (weight loading) |
| Latency sensitivity | TTFT | TPOT |
| Batching behavior | Large input, one pass | Small input, many passes |

When you run both on the same GPU:
- A prefill operation (processing a long prompt) **blocks decode operations** for other requests
- Decode requests see TPOT spikes when a new prefill starts
- You can't independently optimize for compute (prefill) and bandwidth (decode)

```
Same GPU (colocated):
Time →
[Prefill Req A (500ms)]  [Decode Req B] [Decode Req B] [Prefill Req C (300ms)] [Decode Req B]
                         ↑                              ↑
                    TPOT = 30ms                     TPOT = 330ms (blocked by prefill!)
```

---

## 2. The Solution: Separate Prefill and Decode

Run prefill on dedicated **prefill GPUs** and decode on dedicated **decode GPUs**.

```
┌─────────────────────┐     ┌──────────────────────┐
│   Prefill GPUs      │     │   Decode GPUs         │
│   (compute-optimized)│    │   (bandwidth-optimized)│
│                     │     │                        │
│   Process prompts   │────►│   Generate tokens      │
│   Generate KV cache │ KV  │   Stable TPOT          │
│                     │cache│   No prefill spikes     │
└─────────────────────┘ xfer└──────────────────────┘
```

### How it works
1. Request arrives → routed to prefill GPU
2. Prefill GPU processes entire prompt, generates KV cache + first token
3. KV cache transferred to decode GPU (over NVLink, InfiniBand, or network)
4. Decode GPU generates remaining tokens using the transferred KV cache
5. Decode GPU is never interrupted by prefill → stable TPOT

### Benefits
- **Stable TPOT:** Decode GPUs never stall for prefill
- **Independent scaling:** Scale prefill and decode pools separately based on load
- **Hardware specialization:** Could use different GPU types (compute-heavy for prefill, bandwidth-heavy for decode)
- **Better SLO compliance:** Predictable latency for both TTFT and TPOT

### Challenges
- **KV cache transfer overhead:** Moving KV cache between GPUs takes time
- **Increased infrastructure complexity:** Two pools to manage
- **Memory overhead:** KV cache exists on both pools briefly during transfer
- **Network bandwidth:** KV cache for Llama-3 70B at 4096 tokens ≈ 640 MB per request

---

## 3. Who Uses This

### Mooncake (2024)
- KV cache stored in a disaggregated memory pool (CPU memory or distributed storage)
- Prefill nodes write KV cache to pool
- Decode nodes read from pool
- Enables KV cache sharing across requests (prefix caching at cluster level)

### Splitwise (Microsoft, 2024)
- Studied prefill-decode disaggregation on Azure
- Found 20-40% cost savings at same SLO targets
- Prefill on compute-optimized instances, decode on memory-optimized instances

### DistServe (2024)
- Assigns prefill and decode to different GPUs within the same cluster
- Optimizes placement based on workload characteristics
- Demonstrated 2-4x improvement in SLO attainment

### SGLang
- Supports disaggregated prefill as an experimental feature
- Your existing SGLang knowledge connects directly here

---

## 4. Chunked Prefill — A Middle Ground

Instead of full disaggregation, split long prefills into chunks and interleave with decode.

```
Without chunked prefill:
[=== Long Prefill (500ms) ===][Decode][Decode]
                               ↑ All decode requests delayed

With chunked prefill (chunk = 512 tokens):
[Prefill chunk 1][Decode batch][Prefill chunk 2][Decode batch][Prefill chunk 3][Decode batch]
                  ↑ Decode gets regular time slots
```

### How it works
- Set a maximum prefill chunk size (e.g., 512 tokens)
- If prompt > chunk size, process it in multiple chunks
- Between chunks, process pending decode requests
- Reduces maximum TPOT spike from full prefill time to chunk time

### Trade-off
- Increases TTFT (prefill takes longer due to interleaving)
- Reduces TPOT variance (decode is never blocked for long)
- Simpler than full disaggregation — no KV cache transfer needed

### Implementations
- vLLM supports chunked prefill (`--enable-chunked-prefill`)
- SGLang supports it as well
- This is the more common approach in practice (simpler than full disaggregation)

---

## 5. When to Recommend Each Approach

```
Small-medium scale, good enough SLOs?
  → Standard colocated prefill+decode with chunked prefill
  → Simpler, works well for most cases

Strict TPOT SLOs (real-time streaming)?
  → Chunked prefill with small chunks
  → Or prefill-decode disaggregation if budget allows

Very large scale, cost optimization critical?
  → Full prefill-decode disaggregation
  → Different GPU types for each phase
  → 20-40% cost savings per Splitwise paper

Variable workload (bursty prefill)?
  → Disaggregation with independent autoscaling
  → Scale prefill pool during bursts, keep decode pool stable
```

---

## 6. Interview Connections

> "How would you reduce TPOT variance for a real-time financial application?"

**A:** "The main cause of TPOT spikes is prefill operations blocking decode. Three approaches in order of complexity: (1) Chunked prefill — split long prompts into chunks, interleave with decode. Simple, built into vLLM. (2) Priority scheduling — give decode higher priority than prefill when batching. (3) Prefill-decode disaggregation — separate GPU pools, decode is never interrupted. Most aggressive but most effective for strict SLOs."

> "Design a large-scale LLM serving system" (follow-up: how do you handle prefill/decode interference?)

This is an advanced topic that shows you understand beyond basic vLLM deployment.

---

## Self-Test Questions

1. Why do prefill and decode interfere with each other on the same GPU?
2. How does prefill-decode disaggregation solve TPOT spikes?
3. What is the main overhead of disaggregation?
4. How does chunked prefill work? What's the tradeoff?
5. When would you recommend chunked prefill vs full disaggregation?

---

## Notes
```


```
