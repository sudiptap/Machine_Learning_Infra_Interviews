# KV Cache Optimization Techniques

**Priority:** Week 2-3.
**Interview map:** Track A Round 2, Track B Round 3

---

## 1. Why KV Cache Optimization Matters

KV cache is the primary constraint on concurrent requests:

```
Available memory = Total GPU memory - Model weights - Activations
Max concurrent requests = Available memory / (KV cache per request)
Throughput ∝ Max concurrent requests
```

Every optimization that reduces KV cache size directly increases throughput and reduces cost.

---

## 2. KV Cache Quantization

### The idea
Store KV cache in lower precision (INT8 or FP8) instead of FP16.

### Impact
```
FP16 KV cache: 2 bytes per element
INT8 KV cache: 1 byte per element → 2x more concurrent requests
FP8 KV cache:  1 byte per element → same as INT8, native on H100
INT4 KV cache: 0.5 bytes → 4x more concurrent requests (aggressive)
```

### How it works
- Quantize K and V tensors after computing them during prefill
- Store quantized values in cache
- Dequantize when reading during decode attention
- Per-head or per-token scaling factors maintain accuracy

### Quality impact
- INT8 KV cache: minimal quality loss for most models (<0.5% perplexity increase)
- FP8 KV cache: even less loss on H100 (floating point preserves range)
- INT4 KV cache: noticeable quality loss, use with care

### Implementations
- vLLM: `--kv-cache-dtype fp8` or `--kv-cache-dtype int8`
- TRT-LLM: built-in FP8 KV cache support
- This is a "free" throughput boost — always consider it

---

## 3. Sliding Window Attention

### The idea
Instead of attending to ALL previous tokens, only attend to the last W tokens (the "window").

```
Full attention: token 1000 attends to tokens 0-999 (1000 KV entries)
Sliding window (W=4096): token 5000 attends to tokens 901-5000 (4096 KV entries)
```

### Impact on KV cache
- KV cache capped at W tokens regardless of context length
- Memory usage constant, not growing with sequence length
- Enables "infinite" context length in theory

### Quality impact
- Information beyond the window is lost
- Works well for tasks where recent context matters most
- Not suitable when long-range dependencies are critical (e.g., summarizing a 100K document)

### Models using sliding window
- Mistral 7B: window = 4096
- Mixtral: window = 4096

### Hybrid approaches
Some models use sliding window for most layers but full attention for a few layers:
- Captures both local patterns (sliding window layers) and global patterns (full attention layers)
- Jamba, Gemma 2 use this pattern

---

## 4. Token Eviction / Cache Pruning

### The idea
Not all cached tokens are equally important. Evict low-importance tokens to make room.

### Methods

**Attention-score-based eviction:**
- Track which cached tokens receive the most attention over time
- Evict tokens with consistently low attention scores
- Heavy-hitter retention: keep tokens that many queries attend to

**H2O (Heavy-Hitter Oracle):**
- Maintain a "heavy-hitter" set of tokens that accumulate the most attention
- Also keep recent tokens (sliding window for recency)
- Evict everything else
- Reduces KV cache by 5-10x with minimal quality loss on benchmarks

**Scissorhands:**
- Identifies and prunes "unimportant" tokens based on attention patterns
- Uses a pivot token concept to determine importance

### Trade-off
- Reduces memory dramatically
- But: lossy — evicted information is gone
- Quality depends on the task and eviction policy

---

## 5. Multi-Query and Grouped-Query Attention (Architecture Level)

Already covered in `11_attention_variants.md`, but the optimization impact:

```
MHA (32 KV heads) → GQA (8 KV heads):  4x KV cache reduction at architecture level
GQA (8 KV heads)  → MQA (1 KV head):   8x further reduction
```

This is the most impactful "optimization" — but it's a model architecture choice, not an inference optimization. You can't change it after training.

---

## 6. Prefix Caching (System Level)

### The idea
Many requests share the same prefix (system prompt, few-shot examples, RAG context). Cache the KV entries for shared prefixes.

### Impact
```
Without prefix caching:
Request 1 (system prompt + user query): compute KV for ALL tokens
Request 2 (same system prompt + different query): compute KV for ALL tokens again

With prefix caching:
Request 1: compute KV for all tokens, cache system prompt KV
Request 2: reuse cached system prompt KV, only compute for new query tokens
→ 50-90% prefill savings when prefix is long
```

### Implementations
- vLLM: automatic prefix caching (`--enable-prefix-caching`)
- SGLang: RadixAttention (radix tree-based, more efficient for many diverse prefixes)
- TRT-LLM: KV cache reuse

### When it helps most
- Many requests with same system prompt (chatbots)
- RAG with overlapping retrieved documents
- Few-shot prompting with same examples

---

## 7. KV Cache Compression

### Cross-Layer KV Sharing
Some research shows adjacent transformer layers have similar K,V patterns.
- Share KV cache across 2-4 adjacent layers
- 2-4x reduction with small quality impact
- Not yet widely implemented in production

### Low-Rank KV Compression
- Project K,V to lower dimension before caching
- Similar idea to MLA (DeepSeek)
- Reduces cache size proportional to compression ratio
- Adds compute for projection/unprojection

---

## 8. Decision Framework

```
First (free, always do):
  → KV cache quantization (FP8/INT8). 2x savings, minimal quality loss.

Second (if the model supports it):
  → Use a GQA model instead of MHA. 4-8x savings at architecture level.

Third (if requests share prefixes):
  → Enable prefix caching. 50-90% prefill savings.

Fourth (if memory still constrained):
  → Sliding window attention (if model supports it)
  → Token eviction (H2O or similar)
  → More aggressive quantization (INT4 KV cache)

Fifth (if all else fails):
  → Add more GPUs (tensor parallelism) or use a smaller/quantized model
```

---

## Interview Connections

> "A customer has 100 concurrent users on A100 80GB with Llama-3 8B, but they need 200. What do you do?"

**A:** "Step 1: Enable FP8 KV cache — 2x more concurrent requests (free quality). Step 2: Enable prefix caching if requests share system prompts. Step 3: Quantize model weights to INT8 — frees ~8 GB for more KV cache. Step 4: If still not enough, INT4 model weights + FP8 KV cache gives maximum memory for concurrent requests."

> "How does KV cache quantization work? What's the quality impact?"

> "Compare prefix caching in vLLM vs RadixAttention in SGLang."

---

## Self-Test Questions

1. How does KV cache quantization (FP8) double concurrent requests?
2. What is sliding window attention and when does it help?
3. Explain the H2O token eviction strategy.
4. When does prefix caching give the biggest benefit?
5. A customer serves Llama-3 70B on 2x A100 80GB (TP=2) and hits memory limits at 50 concurrent users with 4096 context. Walk through the optimization steps.

---

## Notes
```


```
