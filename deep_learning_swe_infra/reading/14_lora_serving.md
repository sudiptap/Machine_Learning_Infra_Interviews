# LoRA Serving — Multi-Tenant Fine-Tuned Models

**Priority:** Week 3. Important for Track A (SA) enterprise scenarios.
**Interview map:** Track A Round 2 & 4 (multi-tenant design), Track B Round 2

---

## 1. What is LoRA?

Low-Rank Adaptation — a parameter-efficient fine-tuning method.

Instead of fine-tuning ALL model weights (expensive, creates a full model copy), LoRA:
- Freezes the original model weights
- Adds small trainable matrices (adapters) to specific layers
- Adapter size is typically 0.1-1% of original model

```
Original: Y = W × X           (W is huge, e.g., 4096 × 4096)

LoRA:     Y = W × X + (B × A) × X
          where A is (rank × 4096) and B is (4096 × rank)
          rank is small (e.g., 8, 16, 32)

Parameters: W = 16M params (frozen)
            A + B = 2 × rank × 4096 = 65K params (for rank=8)
            → 99.6% reduction in trainable parameters
```

---

## 2. Why LoRA Serving Matters

### The enterprise scenario
A bank has 10 departments, each with a fine-tuned Llama-3 8B:
- Legal team: fine-tuned on legal documents
- Trading desk: fine-tuned on financial analysis
- HR: fine-tuned on company policies

### Without LoRA
- 10 separate full model copies = 10 × 16 GB = 160 GB
- Need multiple GPUs just for the model copies
- Each copy maintains its own KV cache pool

### With LoRA
- 1 base model (16 GB) + 10 LoRA adapters (10 × ~16 MB = 160 MB)
- Total: ~16.2 GB — fits on one GPU!
- Share KV cache memory pool across all adapters
- Dynamically swap adapters per request

---

## 3. How Multi-LoRA Serving Works

```
Request arrives:
  headers: {"model": "llama3-legal-lora"}

┌──────────────────────────────────────┐
│          Base Model (frozen)          │
│    [Weight matrices W₁, W₂, ...]     │
├──────────────────────────────────────┤
│     LoRA Adapter Pool                 │
│  ┌──────────┐ ┌──────────┐          │
│  │ Legal    │ │ Trading  │ ...       │
│  │ (A₁,B₁) │ │ (A₂,B₂) │          │
│  └──────────┘ └──────────┘          │
└──────────────────────────────────────┘

Forward pass for "legal" request:
  Y = W × X + B₁ × A₁ × X
  (base model weights + legal LoRA adapter)
```

### Batching with multiple LoRAs
The challenge: different requests in the same batch may use different adapters.

**Approach 1: Separate batches per adapter**
- Group requests by adapter, run separate batches
- Simple but loses batching efficiency if many adapters with few requests each

**Approach 2: Fused multi-LoRA (S-LoRA approach)**
- Run the base model forward pass for the full batch (shared W × X)
- Run adapter-specific computations in parallel using custom CUDA kernels
- Each request gets its own adapter result added

### S-LoRA (2023)
- Unified paging for LoRA adapters (like PagedAttention for KV cache)
- Custom CUDA kernels for batched LoRA computation
- Can serve thousands of LoRA adapters from a single GPU
- Dynamically loads/unloads adapters as needed

---

## 4. LoRA Serving in Practice

### vLLM
```bash
# Serve with LoRA support
vllm serve meta-llama/Llama-3-8B \
  --enable-lora \
  --lora-modules legal-lora=/path/to/legal-adapter \
                 trading-lora=/path/to/trading-adapter

# Request a specific adapter
curl http://localhost:8000/v1/chat/completions \
  -d '{"model": "legal-lora", "messages": [...]}'
```

### TRT-LLM
- Supports multi-LoRA with runtime adapter switching
- Compiles base model once, loads adapters dynamically

### NIM
- NVIDIA NIM supports LoRA customization
- Upload adapter, reference in API call

---

## 5. Architectural Considerations

### Adapter storage and loading
- Adapters are small (10-100 MB) but you may have hundreds
- Keep hot adapters in GPU memory
- Cold adapters on CPU memory or disk → load on demand
- LRU eviction for adapter cache

### KV cache interaction
- Different adapters change the model's output → KV cache is adapter-specific
- Cannot share KV cache between requests using different adapters
- CAN share KV cache for same prefix + same adapter (prefix caching still works)

### When NOT to use LoRA serving
- If adapters are very different from base model → quality may be poor
- If you need maximum quality → full fine-tuned models may be better
- If you only have 1-2 models → overhead of LoRA serving not worth it

---

## 6. Interview Connections

> "A bank has 15 departments each wanting their own customized LLM. How do you architect this?"

**A:** "Use multi-LoRA serving. One base model (e.g., Llama-3 70B) in GPU memory, 15 LoRA adapters (each ~50-100 MB). Total overhead: <2 GB for all adapters vs 15 × 140 GB for 15 full models. Use vLLM or NIM with LoRA support. Route requests to the correct adapter based on department. Share the base model's compute and KV cache memory pool. This runs on 2 GPUs instead of 30+."

> "How does LoRA batching work when different requests use different adapters?"

> "Design a multi-tenant LLM platform with per-tenant customization."
→ This is essentially the JPMC LLM Suite architecture + LoRA serving

---

## Self-Test Questions

1. How does LoRA reduce fine-tuning cost? What's the typical parameter reduction?
2. Why is multi-LoRA serving more memory-efficient than serving separate models?
3. How does batching work when requests use different LoRA adapters?
4. What is S-LoRA and what problem does it solve?
5. Can you share KV cache between requests using different LoRA adapters? Why or why not?
6. A customer has 100 fine-tuned variants of Llama-3 8B. Compare serving them as full models vs LoRA adapters.

---

## Notes
```


```
