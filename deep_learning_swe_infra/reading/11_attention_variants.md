# Attention Variants — MHA, MQA, GQA, MLA

**Priority:** Week 2. Critical for understanding model architecture tradeoffs.
**Interview map:** Track B Round 3, Track A Round 2

---

## 1. Why Attention Variants Exist

The KV cache is the memory bottleneck of LLM inference. Different attention architectures trade off model quality vs KV cache size.

```
KV cache per token = 2 × num_kv_heads × head_dim × num_layers × dtype_size
```

Reducing `num_kv_heads` directly reduces KV cache size → more concurrent requests → higher throughput.

---

## 2. Multi-Head Attention (MHA) — The Original

From "Attention Is All You Need" (2017).

```
Q: num_heads × head_dim    (e.g., 32 × 128 = 4096)
K: num_heads × head_dim    (same as Q)
V: num_heads × head_dim    (same as Q)
```

- Each head has its own Q, K, V projections
- KV cache stores K and V for ALL heads
- **KV cache per token (Llama-3 8B):** 2 × 32 × 128 × 32 layers × 2 bytes = **524 KB**

### Models using MHA
- GPT-2, GPT-3, OG Llama (original)

---

## 3. Multi-Query Attention (MQA)

Introduced by Noam Shazeer (2019).

```
Q: num_heads × head_dim    (e.g., 32 × 128)
K: 1 × head_dim            (single K head, shared by all Q heads)
V: 1 × head_dim            (single V head, shared by all Q heads)
```

- All query heads share a SINGLE key head and SINGLE value head
- KV cache reduced by `num_heads` factor (e.g., 32x smaller)
- **KV cache per token:** 2 × 1 × 128 × 32 × 2 = **16 KB** (32x reduction!)

### Tradeoff
- Significant KV cache savings
- Some quality degradation (less expressive attention)
- Works well for inference-optimized models

### Models using MQA
- PaLM, Falcon, StarCoder

---

## 4. Grouped-Query Attention (GQA)

Introduced by Ainslie et al. (2023). The sweet spot between MHA and MQA.

```
Q: num_heads × head_dim        (e.g., 32 × 128)
K: num_kv_heads × head_dim     (e.g., 8 × 128 — fewer than Q heads)
V: num_kv_heads × head_dim     (same as K)

Groups: num_heads / num_kv_heads query heads share each KV head
Example: 32 Q heads / 8 KV heads = 4 Q heads per KV group
```

- Middle ground: more KV heads than MQA (better quality) but fewer than MHA (less memory)
- KV cache reduced by `num_heads / num_kv_heads` factor

### KV cache comparison for Llama-3 architecture

| Model | Q Heads | KV Heads | KV Cache/Token | Ratio |
|---|---|---|---|---|
| Llama-3 8B (GQA) | 32 | 8 | 131 KB | 4x smaller than MHA |
| Llama-3 70B (GQA) | 64 | 8 | 164 KB | 8x smaller than MHA |
| Hypothetical MHA 70B | 64 | 64 | 1.3 MB | baseline |
| Hypothetical MQA 70B | 64 | 1 | 20 KB | 64x smaller |

### Models using GQA
- **Llama-3 (all sizes)**, Mistral, Gemma 2, Qwen-2

### Why GQA won
- MQA too aggressive — noticeable quality loss for large models
- GQA with 8 KV heads gives ~95% of MHA quality with 4-8x KV cache reduction
- This is why most modern LLMs use GQA

---

## 5. Multi-Head Latent Attention (MLA) — DeepSeek

Introduced in DeepSeek-V2 (2024).

### The idea
Instead of caching K, V heads directly, compress them into a low-rank latent representation.

```
Standard: cache K (num_kv_heads × head_dim) and V (num_kv_heads × head_dim)

MLA: cache a single latent vector c (compressed_dim)
     At inference: decompress c → K, V using learned projections
     compressed_dim << num_kv_heads × head_dim
```

### How it works (simplified)
1. During prefill: compute K, V normally. Compress into latent `c = W_down × [K; V]`
2. Cache only `c` (much smaller than full K, V)
3. During decode: decompress `K, V = W_up × c`
4. Decompression adds some compute but dramatically reduces KV cache memory

### KV cache comparison
- DeepSeek-V2 (236B params): KV cache is **~93% smaller** than equivalent GQA
- Enables extremely long contexts and high concurrency

### Tradeoff
- Extra compute for decompression
- More complex implementation
- But the memory savings are massive for high-throughput serving

### Models using MLA
- DeepSeek-V2, DeepSeek-V3

---

## 6. Practical Impact on Inference

### Throughput comparison (same GPU, same model quality tier)

```
MHA:  KV cache fills GPU → max ~32 concurrent requests
GQA:  4x smaller cache → max ~128 concurrent requests → ~4x throughput
MQA:  32x smaller cache → max ~1024 concurrent requests → massive throughput
MLA:  ~10-20x smaller cache → high concurrency + long contexts
```

### What this means for serving
- GQA is the standard — most models you'll serve use it
- When a customer asks about serving Llama-3: it uses GQA with 8 KV heads
- KV cache math changes based on the attention variant — always check num_kv_heads

### Corrected KV cache formula
```
KV cache per token = 2 × num_kv_heads × head_dim × num_layers × dtype_size
                     (not num_heads!)
```

---

## 7. Interview Connections

> "How does GQA reduce KV cache size compared to MHA?"

**A:** GQA uses fewer KV heads than query heads. Multiple query heads share each KV head in groups. Llama-3 70B has 64 query heads but only 8 KV heads — 8x KV cache reduction vs MHA, with minimal quality loss.

> "A customer wants to maximize concurrent users on an A100 80GB serving Llama-3 8B. What affects the limit?"

**A:** After model weights (~16 GB FP16), remaining ~64 GB is for KV cache. Llama-3 8B uses GQA with 8 KV heads → ~131 KB per token. At 4096 context: ~524 MB per request. Max concurrent ≈ 64 GB / 524 MB ≈ 122 requests. To increase: quantize KV cache (INT8 → 2x more), reduce context length, or use quantized model to free more memory.

> "What is MLA and why does DeepSeek use it?"

**A:** Multi-Head Latent Attention compresses K,V into a low-rank latent before caching. 93% KV cache reduction vs GQA. Enables extremely long contexts and high concurrency. Tradeoff: extra compute for decompression, but for memory-bound decode this is nearly free.

---

## Self-Test Questions

1. What is the KV cache size per token for MHA vs GQA vs MQA? Give the formula.
2. Llama-3 70B has 64 Q heads and 8 KV heads. What is the GQA group size?
3. Why did GQA win over MQA as the standard?
4. How does MLA compress the KV cache? What's the tradeoff?
5. Calculate max concurrent requests for Llama-3 8B (GQA, 8 KV heads, 128 head_dim, 32 layers, FP16) on A100 80GB at 2048 context length.

---

## Notes
```


```
