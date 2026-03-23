# Speculative Decoding

**Priority:** Read in Week 3–4.
**Interview map:** Track B Round 3

---

## 1. The Problem

LLM decode is sequential — one token at a time. Each token requires a full forward pass through the model (loading all weights from HBM). This is slow, and the GPU is underutilized during decode because it's memory-bound.

**Can we generate multiple tokens per forward pass?**

---

## 2. The Core Idea

Use a small, fast **draft model** to propose multiple tokens, then verify them all at once with the large **target model** in a single forward pass.

```
Without speculative decoding:
Target model: [token1] → [token2] → [token3] → [token4] → [token5]
                5 sequential forward passes

With speculative decoding:
Draft model:  [token1, token2, token3, token4, token5]  ← fast, generate 5 drafts
Target model: [verify all 5 at once]                     ← 1 forward pass (like prefill)
Result:       [token1 ✓, token2 ✓, token3 ✓, token4 ✗, ...]
              Accept first 3, reject token 4, resample from token 4
              → Generated 3 tokens in time of ~1 target forward pass + 5 draft passes
```

### Why verification is cheap
Verifying N draft tokens = running the target model on N tokens simultaneously = prefill-like computation (matrix-matrix multiply, compute-bound). This is much faster than N sequential decode steps (matrix-vector multiply × N, memory-bound).

---

## 3. The Algorithm (Simplified)

```
1. Draft phase:
   - Run draft model for K steps autoregressively
   - Produces K candidate tokens: [d1, d2, ..., dK]
   - Also records draft model probabilities: [p_draft(d1), p_draft(d2), ...]

2. Verify phase:
   - Run target model on ALL K draft tokens in one forward pass
   - Get target model probabilities for each position: [p_target(d1), p_target(d2), ...]

3. Accept/Reject (token by token, left to right):
   - For each token di:
     - If p_target(di) >= p_draft(di): accept always
     - Else: accept with probability p_target(di) / p_draft(di)
     - If rejected: resample from adjusted distribution, STOP accepting further tokens

4. Bonus token:
   - After the last accepted token, sample one more token from the target model
   - This ensures at least 1 new token per round
```

### Key property
The accept/reject scheme guarantees that the output distribution is **identical** to the target model's distribution. Speculative decoding doesn't change the output quality — it only changes the speed.

---

## 4. When It Helps

### Helps when:
- Draft model is much faster than target model (at least 5-10x)
- Draft model has high acceptance rate (agrees with target model often)
- Batch size is small (GPU underutilized during decode → speculative decoding fills the gap)

### Typical speedups:
- 2-3x speedup for greedy/low-temperature sampling
- 1.5-2x for high-temperature/creative sampling (lower acceptance rate)

### Doesn't help when:
- **Large batch sizes:** GPU is already well-utilized during decode; adding draft model overhead doesn't help
- **Draft model quality is poor:** Low acceptance rate → most tokens rejected → no speedup
- **Draft model is too slow:** If draft model takes significant time, the overall speed gain is minimal

---

## 5. Variants

### Standard Speculative Decoding
- Separate draft model (e.g., Llama-3 1B as draft for Llama-3 70B)
- Draft model runs on same GPU or a separate one

### Medusa (Multi-Head Speculative Decoding)
- Instead of a separate draft model, add multiple "Medusa heads" to the target model
- Each head predicts a future token position
- No separate model needed — just extra lightweight heads
- Advantage: no need to find/train a good draft model
- Disadvantage: requires fine-tuning the Medusa heads

### Self-Speculative Decoding (Draft and Verify with Same Model)
- Use early exit from the target model as the draft
- Skip some layers for drafting, use all layers for verification
- No separate model needed, no extra training

### Eagle
- Uses the target model's hidden states to draft tokens
- Trains a lightweight autoregressive head on top of the target model's features
- Higher acceptance rates than Medusa

---

## 6. Interview Answer

> "What is speculative decoding and when does it help? When does it not?"

**Answer:** "Speculative decoding uses a small, fast draft model to propose multiple tokens, then verifies them all at once with the target model in a single forward pass. Verification is cheap because it's a prefill-like operation (matrix-matrix multiply, compute-bound), while sequential decoding is expensive (matrix-vector multiply, memory-bound). It typically gives 2-3x speedup at low batch sizes with a good draft model.

It helps most when: the draft model is much faster than the target, the acceptance rate is high, and the GPU is underutilized (small batch size).

It doesn't help when batch sizes are large (GPU already saturated), the draft model's quality is poor (most tokens rejected), or when the draft model overhead is too high."

---

## Self-Test Questions

1. Why is verifying N tokens faster than generating them one by one?
2. What property does the accept/reject scheme guarantee about output quality?
3. When would you NOT recommend speculative decoding?
4. What is Medusa and how does it differ from standard speculative decoding?
5. A customer runs Llama-3 70B at batch size 1 and wants to improve TPOT. Would you recommend speculative decoding?
6. Same customer now runs at batch size 64. Would speculative decoding still help?

---

## Notes
```
(Take your own notes here)


```
